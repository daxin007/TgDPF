import argparse
import re   
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import logging
import shutil
from timeit import default_timer as timer
import copy
from model import NETLSTM, Transformer
from dataset import *
from utils import *
from configs import *

class AveragePowerCurve(object):
    windspeed_index = 2
    power_index = 3
    
    def __init__(self, datasets, bin_num=50, js_ratio=1.):
        # datasets should be a list of wind turbine data
        self.datasets = datasets
            
        self.bin_num = bin_num
        
        self.js_ratio = js_ratio
        
        # create pc for all data when initialized
        self.create_histograms()
        
        # debug
        # js_record['dataset'].append([self.histograms, self.turbinedata, self.bins])

    def create_histograms(self):
        self.average = torch.zeros(self.datasets[0][0].shape)
        for (turbine_data, _) in self.datasets:
            self.average = turbine_data + self.average
        self.average = self.average/len(self.datasets)
        # remove nan value in average data
        self.average = self.average[~torch.isnan(self.average).any(axis=1)]
        
        self.average = self.average.sub_(self.average.mean(0)).div_(self.average.std(0))
        wind_speed_real = self.average[:, [2]] 
        #self.average[:, [2]] = self.average[:, [4]] 
        
        power = self.average[:, [self.power_index]]
        windspeed = self.average[:, [self.windspeed_index]]
        # print("***************")
        # print(power)
        bins_p = torch.torch.linspace(float(power.min()),
                                        float(power.max()), self.bin_num)
        
        bins_w = torch.torch.linspace(float(windspeed.min()),
                                        float(windspeed.max()), self.bin_num)
        std_w = windspeed.std()
        ratio_p = power.abs().mean()
        
        mean_p = power.mean()
        
        # statistics
        self.bins = [bins_p, bins_w, ratio_p, std_w, mean_p]
        print(self.bins)
        
        pc = self.power_curve(power, windspeed, bins_p, ratio_p, self.bin_num)
        # self.average[:, [2]] = self.average[:, [4]]
        
        self.average_pc = pc
            
    def dataset_js(self, model):
        model.train()
        
        data_input, _ = time_ahead(self.average, training=False)
        
        pc = self.average_pc.to(DEVICE)
        data_input = data_input.to(DEVICE)
        # make prediction
        power_predict, _ = model(data_input.reshape(1, len(data_input), -1), training=False)
        # power_predict = model(data_input.reshape(1, len(data_input), -1), training=False)
        windspeed = data_input[:, [1]]
        
        # data clip
        power_predict = power_predict[OFFSET-TIME_AHEAD:, :]
        windspeed = windspeed[OFFSET-TIME_AHEAD:, :]
        
        pc0 = self.power_curve(
            power_predict, 
            windspeed, 
            self.bins[0], 
            self.bins[2],
            self.bin_num)
        
        pc_record.append([pc0.cpu(), pc.cpu()])
        # debug
        # js_record['pc'].append([power_predict, windspeed, pc0, pc, js_i])
                            
        js_loss = js_div(pc0, pc)
        
        return js_loss * self.js_ratio
        # if self.js_ratio:          
        #     return js_loss
        # else:
        #     return torch.tensor(0)
        
    def data_checker(self, data):
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        return data
        
    def power_curve(self, power, windspeed, bins_p, ratio_p, bins):
        power = self.data_checker(power)
        windspeed  = self.data_checker(windspeed)
        
        power = power.unsqueeze(0)
        windspeed = windspeed.unsqueeze(0) 
        
        # bins_p = torch.torch.linspace(float(power.min()),
        #                                 float(power.max()), self.bin_num)
        bins_p = bins_p.to(power.device)
        
        bins_w = torch.torch.linspace(float(windspeed.min()),
                                        float(windspeed.max()), self.bin_num)
        bins_w = bins_w.to(windspeed.device)
        std_w = windspeed.std()
        # power = power - power.mean()
        # self.js_ratio = abs(1 - power.std())+0.5
        # power = power * (bins_p.max() / power.max())
        # print('power mean: ', power.abs().mean().item(), power.mean().item(), ratio_p)
        
        
        # make the predict power average and scaler fit the real power
        # power = power * (ratio_p / power.abs().mean())
        
        # power = power - power.mean()

            
        # brandwidth_sigma = torch.pow(torch.tensor(bins, dtype=torch.float32), -1/5)*0.07
        # 2022-5-27 0.07
        brandwidth_sigma = torch.pow(torch.tensor(bins, dtype=torch.float32), -1/5)*0.1

        if power.std() < 1e-4 or power.mean() < bins_p.min() or power.mean() > bins_p.max():
            self.js_ratio = 0.
        else:
            self.js_ratio = 1.
        # set js ratio dynamicly
        # self.js_ratio = abs(1 - power.std())
        # print(power.std())
        brandwidth_p = power.std()*brandwidth_sigma 

        brandwidth_w = std_w*brandwidth_sigma
            
        _, kernel_p = marginal_pdf(power, bins_p, brandwidth_p, 1e-10)
        _, kernel_w = marginal_pdf(windspeed, bins_w, brandwidth_w, 1e-10)
        pc = joint_pdf(kernel_p, kernel_w)
        return pc

def trainer(dataloader, model, criterion, optimizer, awl, powercurve, epoch, record, logger, jsmode=True, jsonly=False):
    
    # switch to train mode
    model.train(True)
    
    t = tqdm(dataloader, total=int(len(dataloader)))
    
    t.set_description(f'Epoch {epoch+1}')
    
    for _, (input_, target) in enumerate(t):
        
        optimizer.zero_grad()
        
        if jsmode:
            js = powercurve.dataset_js(model)
            record.update('js_loss', js.item())
            record.update('js_ratio', powercurve.js_ratio)
            print(f'js_loss: {js.item()}, js_ratio: {powercurve.js_ratio}')
            # if powercurve.js_ratio == 0:
            #     optimizer.zero_grad()
            #     model.zero_grad()
            
            if jsonly:
                logger.info('only use js loss')
                loss = js
                record.update('mse', 0)
                
            else:
                logger.info('use js and mse loss')
                target = target.reshape(-1, 1)
                
                target = target.to(DEVICE)
                input_ = input_.to(DEVICE)
                
                # print(input_.shape)
                output, _ = model(input_)
                # output = model(input_)
                loss_local = criterion(output, target)

                record.update('mse', loss_local.item())
                
                # if powercurve.js_ratio > 0:
                       
                #     loss = loss_local + js
                # else:
                #     loss = loss_local
                
                loss = loss_local + js
                # loss = awl(loss_local, js)


        else:
            logger.info('use mse loss')
            target = target.reshape(-1, 1)
            
            target = target.to(DEVICE)
            input_ = input_.to(DEVICE)
            
            output, _ = model(input_)
            #output = model(input_)
            #print(output.shape)
            # output = output[:, -TIME_AHEAD:, :].reshape(-1,1)
            loss = criterion(output, target)
            
            record.update('mse', loss.item())


        loss.backward()
        # # debug
        # js_record['parameters'].append(model.state_dict()['lstm.weight_ih_l0'].detach().cpu().T)
        # js_record['model'].append(copy.deepcopy(model))
    
        # js_record['loss'].append(loss)
        
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        # save record
        # with open(f'js_record11.pkl', 'wb') as f:
        #     pickle.dump(js_record, f)
                
        # try:
        #     for name, param in model.named_parameters():
        #         print(name, torch.isfinite(param.grad).all())
        # except (RuntimeError, TypeError, NameError):
        #     print('false')
                
        t.set_postfix(loss='{:05.3f}'.format(loss))
        record.update('loss', loss.item())
        
        t.update()

 
def evaluater(model, test_input):
    model.eval()
    with torch.no_grad():
        predict, _ = model(test_input.reshape(1, len(test_input), -1), training=False)
        # predict = model(test_input.reshape(1, len(test_input), -1))
        # predict = predict.reshape(-1, 1)
    return predict


def test_noise_rate(testdataset, model, criterion, test_noise, logger):
    record = {}
    if test_noise:
        logger.info('test different noise rate')
        for noise_rate in [0, 0.1, 0.3, 0.5, 0.7]:
            testdataset.set_noise_rate(noise_rate)
            logger.info(f'set testdatasedt noise rate to {noise_rate}')
            # record[noise_rate] = tester(model, criterion, testdataset, logger)
            record[noise_rate] = nan_seg_tester(model, criterion, testdataset, logger)
    else:
        record[0] = nan_seg_tester(model, criterion, testdataset, logger)
            
    return record



def nan_seg_tester(model, criterion, nan_seg_dataset, logger):
    model.eval()
    
    record = {}
    
    for i, (testset, name) in enumerate(nan_seg_dataset):
        full_test_target = []
        full_predict = []
        for data_seg in testset:
            test_input, test_target = data_seg
            test_input = test_input.to(DEVICE)
            test_target = test_target.to(DEVICE)
            
            predict = evaluater(model, test_input)
            predict = predict[OFFSET-TIME_AHEAD:, :]
            
            full_test_target.append(test_target)
            full_predict.append(predict)
        
        full_test_target = torch.cat(full_test_target)
        full_predict = torch.cat(full_predict)
        
        test_loss = criterion(full_predict, full_test_target)
        
        logger.info(f'{name}: {test_loss.item()}')
        
        test_record = {
            'index': i,
            'pred': full_predict,
            'targ': full_test_target,
            'loss': test_loss.item(),
        }
        
        record[name] = test_record
        
    return record

def full_mode(
    experiment_id=None,
    start_date=None,
    js_mode=False,
    noise_rate=0,
    batch_size=500,
    epochs=100,
    train_len=10,
    test_noise=False,
    train_noise=False,
    logger=None,
    evaluate=False,
    js_only=False,
    js_ratio=1.,
    load_exist=False,
    ):
    
    start_epoch = 0
    record = {
        'checkpoint': {},
        'testloss': {},
    }
    
    # awl = AutomaticWeightedLoss(2)
    awl = None
    # testdataset = TestDataset(root_dir, start_date=start_date, noise_rate=noise_rate)
    testdataset = NanSegDataset(ROOT_DIR, start_date=start_date, noise_rate=noise_rate)
    criterion = nn.MSELoss()
    
    model = NETLSTM(
        dim_in=3,
        dim_hidden=20,
        dim_out=1,
        num_layer=2,
        dropout=0.3)
    model.to(DEVICE)
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logger.info(f'set learning rate to {lr}')
    
    if evaluate:
        parameters = load_model(experiment_id, start_epoch)
        
        model.load_state_dict(parameters['state_dict'])
        
        test_noise_rate(testdataset, model, criterion, test_noise, logger)
        
    else:
        logger.info('train begin')
        
        train_record = Record()
        
        if load_exist:
            logger.info('load exist model')
            parameters = load_model(experiment_id, start_epoch)
            
            model.load_state_dict(parameters['state_dict'])
            
            optimizer.load_state_dict(parameters['optimizer'])
            
            backup_record(experiment_id)
        
        model.zero_grad()

        if train_noise:
            logger.info(f'train_noise mode on, set noise to {noise_rate}')
            training_dataset = SeqDataset_KFold(
                ROOT_DIR,
                file_list = FILE_LIST,
                split_date = start_date,
                train_len = train_len,
                noise_rate=noise_rate
            )
        else:
            logger.info('remove train noise')
            training_dataset = SeqDataset_KFold(
                ROOT_DIR, 
                file_list = FILE_LIST,
                split_date = start_date,
                train_len = train_len,
                noise_rate=0
            )
        
        # powercurve = PowerCurve(list(training_dataset.datasets()), js_ratio=js_ratio)
        powercurve = AveragePowerCurve(list(training_dataset.datasets()), js_ratio=js_ratio)
        
        dataloader = DataLoader(
            training_dataset,
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            num_workers=4)
        for epoch in range(epochs):
            epoch = epoch + start_epoch
            logger.info(f'running epoch {epoch}')
            
            # training
            start = timer()
            trainer(dataloader, model, criterion, optimizer, awl,
                    powercurve, epoch, train_record, logger, jsmode=js_mode, jsonly=js_only)
            end = timer()
            logger.info(f'epoch {epoch}: func [trainer] cost {end-start}')
            test_record = test_noise_rate(testdataset, model, criterion, test_noise, logger)
            
            # save checkpoint
            # print(model.state_dict())
            record['checkpoint'][epoch] = {
                'state_dict': copy.deepcopy(model.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()),
            }
            
            record['testloss'][epoch] = test_record

        logger.info('training complete')
        
        # wirite record
        record['epochs'] = epochs + start_epoch
        record['train_loss'] = train_record
        record['model'] = model
        record['pc'] = pc_record
        record['scaler'] = testdataset.scalers

        logger.info('save record')
        with open(f'record/{experiment_id}.pkl', 'wb') as f:
            pickle.dump(record, f)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

        # with open(f'js_record6.pkl', 'wb') as f:
        #     pickle.dump(js_record, f)

def main():
    parser = argparse.ArgumentParser(description='Wind Power Forecast')
    parser.add_argument('--batch_size', type=int, default=300, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epoch to train (default: 10)')
    parser.add_argument('--seed', type=int, default=4444, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--id', type=str, default='lstm_origin', metavar='N',
                        help='experiment id (default: 0)')
    parser.add_argument('--start', type=str, default='2020-11-01', metavar='XXXX-XX-XX',
                        help='start date (default: 2020-11-01)')
    parser.add_argument('--train_len', type=int, default=10, metavar='N',
                        help='training date legenth')
    parser.add_argument('--js_mode', action='store_true', default=False,
                        help='enable js loss')
    parser.add_argument('--noise_rate', type=float, default=0, metavar='V',
                        help='noise for target')
    parser.add_argument('--test_noise', action='store_true', default=False,
                        help='add test noise')
    parser.add_argument('--train_noise', action='store_true', default=False,
                        help='add training noise')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate only')
    parser.add_argument('--js_only', action='store_true', default=False,
                        help='only js loss')
    parser.add_argument('--js_ratio', type=float, default=1.0, metavar='V',
                        help='js loss ratio')
    parser.add_argument('--load_exist', action='store_true', default=False,
                        help='load exist model')

    # os.chdir(r'G:\风能\DT')
    if not os.path.exists('record'):
        os.mkdir('record')
        

    args = parser.parse_args()
    
    logging.basicConfig(filename=f'log/log-{args.id}.log',
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)
    
    logger = logging.getLogger(f'Experiment-{args.id}')
    
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # write log
    print(args.id)
    logger.info(f'Experiment id: {args.id}')
    logger.info(f"batch_size:\t{args.batch_size}\n"
                f"epochs:\t{args.epochs}\n"
                f"seed:\t{args.seed}\n"
                f"start:\t{args.start}\n"
                f"js mode:\t{args.js_mode}\n"
                f"js only:\t{args.js_only}\n"
                f"noise rate:\t{args.noise_rate}\n"
                f"test noise:\t{args.test_noise}\n"
                f"train_len:\t{args.train_len}\n"
                f"train noise:\t{args.train_noise}\n")
    logger.info(f'js ratio: {args.js_ratio}')
    
    full_mode(experiment_id=args.id,
              start_date=args.start,
              js_mode=args.js_mode,
              noise_rate=args.noise_rate,
              batch_size=args.batch_size,
              epochs=args.epochs,
              test_noise=args.test_noise,
              train_noise=args.train_noise,
              train_len=args.train_len,
              logger=logger,
              js_only=args.js_only,
              js_ratio=args.js_ratio,
              load_exist=args.load_exist)

 
if __name__ == '__main__':
    main()