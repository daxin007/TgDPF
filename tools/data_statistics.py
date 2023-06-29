import pickle 
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('dataset_summary.pdf')
import matplotlib
#matplotlib.rc("font",family='SimHei')

file_dir = '江苏十分钟数据'

def read_data(filepath):
    df = pd.read_csv(filepath, encoding='gbk')
    df['时间'] = pd.to_datetime(df['时间'])
    df.set_index(['时间'], inplace=True)
    full_perid = pd.date_range(start='20200101', end='20210101',freq='10min', closed='left')
    df = pd.DataFrame(df, index=full_perid)
    return df

def data_clean(df):
    df = df.interpolate(limit=1)
    df[df<0] = 0
    return df

def data_loader(file_dir):
    for data_file in os.listdir(file_dir):
        if data_file.endswith('csv'):
            file_path = os.path.join(file_dir, data_file)
            data = read_data(file_path)
            data = data_clean(data)
            data = data[~np.isnan(data).any(axis=1)]
            yield data_file, data
 
 
def dataset_average(file_dir):
    dataset = []
    for data_file in os.listdr(file_dir):
        if data_file.endswith('csv'):
            file_path = os.path.join(file_dir, data_file)
            data = read_data(file_path)
            dataset.append(data)
            
            
            
def box_plot(data, title, xticks):
    fig, ax = plt.subplots(figsize=(16,5))
    ax.set_title(title)
    ax.violinplot(data, showmeans=False, showmedians=True)
    plt.setp(ax, xticks=[i+1 for i in range(len(xticks))],
             xticklabels=xticks)
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    pp.savefig(fig)
    
               
def dataset_statistics(data_sources):
    head = ['桨距角', '发电机转速', '平均风速', '有功功率']
    head_eng = ['angle', 'generator speed', 'wind speed', 'power']
    record = {i:[] for i in head}
    file_names = []
    for file_name, data in data_sources:
        file_names.append(file_name.split('.')[0])
        for column in head:
            record[column].append(data[column].values)
    for i, name in enumerate(head):
        box_plot(record[name], head_eng[i], file_names)
    pp.close()  
    # with open('data_statistics.pkl', 'wb') as f:
    #     pickle.dump(record, f)
     


dataset_statistics(data_loader(file_dir))

