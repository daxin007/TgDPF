from math import exp
import pickle
from re import T
from typing import Type
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from train_dayahead import *

filenames = get_file_list('江苏十分钟数据', r'.*csv')

summary_dir = 'record'
os.environ['NUMEXPR_MAX_THREADS'] = '32'


def load_record(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pc_plot(pc0, pc1, title, canvas, num_bins=50):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle(title)
    axs[0].imshow(pc0.reshape(num_bins, num_bins))
    axs[0].set_title('model')
    axs[1].imshow(pc1.reshape(num_bins, num_bins))
    axs[1].set_title('true')
    plt.tight_layout()
    canvas.savefig(fig)
    plt.close()


def pc_record(experiment_id):
    pp = PdfPages(f'record/{experiment_id}/pc_summary.pdf')
    data = load_record(f'record/{experiment_id}.pkl')
    pc_record_index = list(map(lambda x: x * 25 + 0, list(range(len(data['pc']) // 25))))
    for i in pc_record_index:
        pc = data['pc'][i]
        pc_plot(pc[0].detach().numpy(), pc[1].detach().numpy(), i, pp)
    pp.close()


def plot(pred, targ, n, title, experiment_id, pp):
    fig, axs = plt.subplots(n, 1, figsize=(16, 9))
    fig.suptitle(title)
    for i, (pred, targ) in enumerate(zip(np.array_split(pred, n), np.array_split(targ, n))):
        axs[i].plot(pred, c='r', label='pred')
        axs[i].plot(targ, c='b', label='targ')
        axs[i].legend()
    plt.tight_layout()
    pp.savefig(fig)
    plt.savefig(f'{summary_dir}/{experiment_id}/img/{title}.png')
    plt.close()


def compare_plot(pred1, pred2, targ, n, title, experiment_id, pp):
    fig, axs = plt.subplots(n, 1, figsize=(16, 9))
    fig.suptitle(title)
    for i, (pred1, pred2, targ) in enumerate(
            zip(np.array_split(pred1, n), np.array_split(pred2, n), np.array_split(targ, n))):
        axs[i].plot(pred1, c='r', label='pred1')
        axs[i].plot(pred2, c='g', label='pred2')
        axs[i].plot(targ, c='b', label='targ')
        axs[i].legend()
    plt.tight_layout()
    pp.savefig(fig)
    plt.savefig(f'{summary_dir}/{experiment_id}/img/{title}.png')


def summary(experiment_id='112', makeplot=True, epoch=None):
    if not os.path.exists(f'{summary_dir}/{experiment_id}'):
        os.mkdir(f'{summary_dir}/{experiment_id}')

    if not os.path.exists(f'{summary_dir}/{experiment_id}/img'):
        os.mkdir(f'{summary_dir}/{experiment_id}/img')

    data = load_record(f'{summary_dir}/{experiment_id}.pkl')

    if epoch:
        record_epoch = epoch - 1
    else:
        record_epoch = data['epochs'] - 1

    for noisy_rate, test_record in data['testloss'][record_epoch].items():
        record = []
        pp = PdfPages(f'{summary_dir}/{experiment_id}/{experiment_id}_{noisy_rate}_summary.pdf')

        for filename, record_data in test_record.items():
            pred = record_data['pred'].cpu()

            targ = record_data['targ'].cpu()
            scaler = data['scaler'][filename]
            # mean = scaler.mean_[-1]
            # std = np.sqrt(scaler.var_[-1])
            #
            mean = scaler[0][-1]
            std = scaler[1][-1]

            pred = pred * std + mean
            targ = targ * std + mean
            pred = pred.flatten().numpy()
            targ = targ.flatten().numpy()
            pres = 1 - np.abs(pred - targ) / np.max(targ)
            if data['train_loss'].data('js_loss'):
                js_loss = data['train_loss'].data('js_loss')[-1]
            else:
                js_loss = 0
            record.append(
                [filename, record_data['loss'], data['train_loss'].data('loss')[-1], data['train_loss'].data('mse')[-1],
                 js_loss, np.mean(pres), np.median(pres)])
            if makeplot:
                plot(record_data['pred'].cpu().numpy(), record_data['targ'].cpu().numpy(), 5,
                     str(noisy_rate) + filename, experiment_id, pp)

        df = pd.DataFrame(record, columns=['name', 'test_loss', 'train_loss', 'mse_loss', 'js_loss', 'precision',
                                           'precision_median'])
        df.set_index('name', inplace=True)
        df.loc['mean'] = df.apply(lambda x: x.mean())
        df.loc['median'] = df.apply(lambda x: x.median())
        df.loc['std'] = df.apply(lambda x: x.std())
        df.to_excel(f'{summary_dir}/{experiment_id}/epoch{record_epoch}_{experiment_id}_{noisy_rate}_summary.xlsx')
        pp.close()


def get_report(filename):
    df = pd.read_excel(filename)
    return df.iloc[:-3, 1:], df.iloc[-3:, 1:]


def make_summary(item, epoch):
    if not isinstance(item, list):
        item = [item]
    for i in item:
        # items = ['mse', 'js_loss']
        items = ['js_loss', 'mse']
        summary(experiment_id=str(i), makeplot=True, epoch=epoch)
        pc_record(str(i))
        # get_loss([i], items, save_loc=f'record/{i}/loss.csv')
        print('.')
        # compare_loss(f'record/{i}/loss.csv', items, title=str(i), save_loc=f'record/{i}/loss.png')


def get_loss(item, types: list, save_loc='loss.csv'):
    record = []
    try:
        item = iter(item)
    except TypeError:
        item = [item]
    for i in item:
        data = load_record(f'{summary_dir}/{i}.pkl')
        for item in types:
            loss_data = data['train_loss'].data(item)
            # if item == 'js_loss':
            #     loss_data = [j*10 for j in loss_data]
            record.append(loss_data)
    np.savetxt(save_loc, np.array(record), delimiter=',')


def merge_report(filename, items, info, epoch=20):
    tmp = []
    reports = []
    for item in items:
        experiment_report_name = f'record/{item}/epoch{epoch}_{item}_{info}_summary.xlsx'
        df = pd.read_excel(experiment_report_name)
        reports.append(df.iloc[:-3, 1:])
        tmp.append(df.iloc[-3:, 1:])
    report_name = f'report/{filename}-{info}.xlsx'
    report = pd.concat(reports, ignore_index=True)
    report.loc['mean'] = report.apply(lambda x: x.mean())
    report.loc['median'] = report.apply(lambda x: x.median())
    report.loc['std'] = report.apply(lambda x: x.std())
    report.to_excel(report_name)
    tmp = pd.concat(tmp, ignore_index=True)
    # tmp = report.iloc[-3:, :]
    tmp.to_excel(f'report/{filename}-{info}_s.xlsx')
    return tmp


def main(start=2000, count=20, step=5, info=None, prefix=None, noisy_rate=None):
    num = int(count / step)
    tmp = []
    reports = []
    # seed = ['0', '1111','2222','3333','4444']
    for i in range(0, count):
        filename = f'record/{i + start}/{i + start}_{noisy_rate}_summary.xlsx'
        df, a1 = get_report(filename)
        reports.append(df)
        tmp.append(a1)
        print(f'add record/{i + start}/{i + start}_{noisy_rate}_summary.xlsx to report')
        if i % num == num - 1:
            print(f'[{start + i}] running {info[i // num]}')
            report_name = f'report/{prefix}_{info[i // num]}_{noisy_rate}.xlsx'
            report = pd.concat(reports, ignore_index=True)
            report.loc['mean'] = report.apply(lambda x: x.mean())
            report.loc['median'] = report.apply(lambda x: x.median())
            report.loc['std'] = report.apply(lambda x: x.std())
            report.to_excel(report_name)
            tmp = pd.concat(tmp, ignore_index=True)
            tmp.to_excel(f'report/{prefix}_{info[i // num]}_{noisy_rate}_all.xlsx')
            reports = []
            tmp = []


def get_testloss(name, data, noisy_rate=0.7):
    record = []
    for i in range(data['epochs']):
        record.append(data['testloss'][i][noisy_rate][name]['loss'])
    return record


def compare_loss(filename, items, title=None, save_loc='loss.png', show=False):
    data = np.loadtxt(filename, delimiter=',')
    color1 = plt.cm.rainbow(np.linspace(0, 0.1, len(items) // 2))
    color2 = plt.cm.rainbow(np.linspace(0.6, 0.7, len(items) // 2))
    color = np.vstack([color1, color2])
    for i in range(len(items)):
        plt.plot(data[i, :], label=items[i], c=color[i])
    if title:
        plt.title(title)
    plt.legend()
    # plt.tight_layout()
    plt.savefig(save_loc)
    plt.close()
    if show:
        plt.show()


def addtxt(filename, info):
    with open(filename, 'a') as f:
        f.write(f'{info}\n')


def run(start=4125, prefix=0.01):
    start_date = [
        '2020-07-01',  # 4 months
        '2020-06-01',  # 5 months
        '2020-05-01',  # 6 months
        '2020-04-01',  # 7 months
        '2020-03-01',  # 8 months
        '2020-02-01',  # 9 months
        '2020-01-01',  # 10 months
    ]
    kfold_start_date = [datetime.date(2020, m, 1).strftime('%Y-%m-%d') for m in (1, 3, 5, 7, 9, 11)]
    # noisy_rate = [0, 0.1, 0.3, 0.5, 0.7]
    noisy_rate = [0]
    epochs = list(range(10, 101, 10))
    # start = 4125
    for epoch in epochs:
        # make_summary([3529], epoch)
        make_summary(list(range(start, start + 5)), epoch)
    start = start - 25
    tmp_file = 'report/tmp.csv'
    addtxt(tmp_file, start)
    for epoch in epochs:
        for p in range(5, 6):
            record_mean = []
            record_median = []
            info = f'2022-9-27-js-{start}-6fold-10months-{epoch}epoch-{kfold_start_date[p]}'
            for i in noisy_rate:
                report = merge_report(info, [start + p * 5 + j for j in [0, 1, 2, 3, 4]], i, epoch=epoch - 1)
                mean = report.test_loss[[i * 3 for i in range(5)]].to_list()
                median = report.test_loss[[i * 3 + 1 for i in range(5)]].to_list()
                # mean = report.test_loss[[0]].tolist()
                # median = report.test_loss[[1]].tolist()
                record_mean.append(mean)
                record_median.append(median)
            record_all = np.vstack([np.array(record_mean).T, np.array(record_median).T])
            record_all = pd.DataFrame(record_all, columns=noisy_rate)
            record_all.loc['median_mean'] = record_all.iloc[-5:, :].mean()
            addtxt(tmp_file, ','.join(str(i) for i in record_all.loc['median_mean']))
            record_all.loc['mean_mean'] = record_all.iloc[:5, :].mean()
            record_all_name = f'report/a_{info}.xlsx'
            record_all.to_excel(record_all_name)

    # summary(experiment_id='8')
    # items = list(range(2010, 2014))
    # merge_report(f'2022-1-15-js-2020-11-01', [7015, 7016, 7017, 7018, 7019], 0.7)
    # make_summary([6015])
    # get_loss([1016, 1017, 1018, 1019], ['mse', 'js_loss'])
    # for i in [0, 0.1, 0.3, 0.5, 0.7]:
    #     main(start=2000, count=20, step=4, info=start_date, prefix='lstm_start_date', noisy_rate=i)
    # main(start=2000, count=25, step=5, info=start_date, prefix='lstm_start_date')


if __name__ == '__main__':
    # for i, j in zip([4025, 4125, 4225, 4325, 4425], [0.015, 0.01, 0.05, 0.1, 0.3]):
    #     run(i, prefix=j)
    # run(start=2825, prefix=0)
    for epoch in list(range(10, 51, 10)):
        print(epoch)
        make_summary([99999], epoch)