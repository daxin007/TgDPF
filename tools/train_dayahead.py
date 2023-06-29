from email.policy import default
import numpy as np
import torch.utils.data
import pandas as pd
from sklearn import preprocessing
import os
from torch.autograd import Variable
import torch.nn.functional as F  
import argparse
import re   
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import logging
import shutil
from timeit import default_timer as timer
import datetime
import sys
# sys.path.append("D:\gjx\cyt\风能\DT\江苏十分钟数据")


os.environ['NUMEXPR_MAX_THREADS'] = '16'

torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda') 
    
else:
    DEVICE = torch.device('cpu')
    
def get_file_list(file_dir, pattern):
    pattern = re.compile(pattern)
    return sorted([filename for filename in os.listdir(file_dir) if pattern.match(filename)])


def read_data(filepath):
    df = pd.read_csv(filepath, encoding='gbk')
    df['时间'] = pd.to_datetime(df['时间'])
    df.set_index(['时间'], inplace=True)
    full_perid = pd.date_range(start='20200101', end='20210101',freq='10min', closed='left')
    df = pd.DataFrame(df, index=full_perid)
    # with open('synthetic_wind_speed/task1.pkl', 'rb') as f:
    #     df2 = pickle.load(f)
    df2 = pd.read_csv('synthetic_wind_speed/task1.csv')
    df2.rename(columns={'Unnamed: 0':'时间'},inplace=True)
    df2['时间'] = pd.to_datetime(df2['时间'])
    df2.set_index(['时间'], inplace=True)
    # print(df)
    # print("################")
    # print(df2)
    df['wp'] = df2['wind_speed']
    return df


def make_dataset(files):
    dataset_path = 'dataset2.pkl'
    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = []
        for file in files:
            data = read_data(file)
            dataset.append(data)
        # with open(dataset_path, 'wb') as f:
        #     pickle.dump(dataset, f)
    return dataset


SPLIT_DATE = '2020-11-01'

ROOT_DIR = '江苏十分钟数据'
FILE_LIST = get_file_list(ROOT_DIR, r'.*csv')
# print(FILE_LIST)
FILE_LIST = ['江苏十分钟数据/'+f for f in FILE_LIST]
if not os.path.exists('record'):
    os.mkdir('record')
    
DATASET = make_dataset(FILE_LIST)

TIME_AHEAD = 6*24
WINDOW_SIZE = 5*6*24
STEP = 6*24
OFFSET = WINDOW_SIZE-TIME_AHEAD


Tensor = torch.tensor
pc_record = []

js_record = {
    'js': [],
    'pc': [],
    'dataset': [],
    'model': [],
    'loss':[],
    'mse': [],
    'parameters': [],
    'brandwidth': [],
} 


class  NETLSTM(nn.Module):

    def __init__(self, dim_in=None, dim_hidden=None, dim_out=None,
                 num_layer=None, dropout=None, **kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_layer = num_layer
        self.dropout = dropout
        self.dim_out = dim_out
        self.lstm = nn.LSTM(self.dim_in,
                            self.dim_hidden,
                            self.num_layer,
                            batch_first = True,
                            dropout = self.dropout)
        self.fc2 = nn.Linear(self.dim_hidden, int(self.dim_hidden/2))
        self.fc3 = nn.Linear(int(self.dim_hidden/2), self.dim_out)
        self.fc4 = nn.Linear(int(self.dim_hidden/2), int(self.dim_hidden/2))
        self.bn = nn.BatchNorm1d(int(self.dim_hidden / 2))
        self.device = DEVICE
        
    def forward(self, x, hs=None, training=True):  
        if hs is None:
            h = Variable(torch.zeros(self.num_layer, len(x), self.dim_hidden, device=self.device))
            v = Variable(torch.zeros(self.num_layer, len(x), self.dim_hidden, device=self.device))
            hs = (h, v)
        out, hs_0 = self.lstm(x, hs)
        if training:
            out = out[:, -TIME_AHEAD:, :]
        out = out.contiguous()
        out = out.view(-1, self.dim_hidden)                
        # out = nn.functional.relu(self.bn(self.fc2(out)))
        out = nn.functional.relu(self.fc2(out))
        # out = nn.functional.tanh(self.fc2(out))
        # out = nn.functional.relu(self.fc4(out))
        out = self.fc3(out)
        return out, hs_0
    
    def set_device(self, device):
        self.device = device


def data_clean(df):
    df = df.interpolate(limit=1)
    df[df<0] = 0
    return df


def nan_seg(df):
    # return data segmentation seplit by continuous nan value
    # data for checking nan value
    # arr = df[column_name].values
    # get value position where nan appear
    # thre first element must be True so add True before the result
    # Select all rows with NaN under the entire DataFrame
    arr = df.isnull().any(axis=1).values
    # and state change point
    nan_pos = np.r_[1, np.diff(arr), 1]
    start = 0
    for i, flag in enumerate(nan_pos):
        if flag and i>0:
            if not arr[start]:
                yield df.iloc[start:i]
            start = i


def marginal_pdf(values, bins, a, b = 1e-10):
    
    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / a).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + b
    pdf = pdf / normalization

    return pdf, kernel_values + torch.tensor(1e-10)


def joint_pdf(kernel_values1, kernel_values2):

    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
    normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1)
    pdf = joint_kernel_values / normalization

    return pdf



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int,the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

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
        
        bins_p = bins_p.to(power.device)
        
        bins_w = torch.torch.linspace(float(windspeed.min()),
                                        float(windspeed.max()), self.bin_num)
        bins_w = bins_w.to(windspeed.device)
        std_w = windspeed.std()
        # power = power - power.mean()
        # self.js_ratio = abs(1 - power.std())+0.5
        # power = power * (bins_p.max() / power.max())
        print('power mean: ', power.abs().mean().item(), power.mean().item(), ratio_p)
        
        
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

class PowerCurve(object):
    windspeed_index = 2
    power_index = 3
    
    def __init__(self, datasets, bin_num=50, js_ratio=1.):
        # datasets should be a list of wind turbine data
        self.datasets = datasets
        self.turbinedata = {}
        
        self.histograms = {}
        
        self.js_ratio = js_ratio
        
        self.bin_num = bin_num
        self.bins = {}
        
        # create pc for all data when initialized
        self.create_histograms()
        
        # debug
        # js_record['dataset'].append([self.histograms, self.turbinedata, self.bins])

    def create_histograms(self):
        for _, (turbine_data, file_id) in enumerate(self.datasets):
            turbine_data = turbine_data[~torch.isnan(turbine_data).any(axis=1)]
            power = turbine_data[:, [self.power_index]]
            windspeed = turbine_data[:, [self.windspeed_index]]
            self.turbinedata[file_id] = turbine_data
            bins_p = torch.torch.linspace(float(power.min()),
                                          float(power.max()), self.bin_num)
            bins_w = torch.torch.linspace(float(windspeed.min()),
                                          float(windspeed.max()), self.bin_num)
            
            std_p = power.std()
            std_w = windspeed.std()
            
            mean_p = power.mean()
            
            self.bins[file_id] = [bins_p, bins_w, std_p, std_w, mean_p]
            
            pc = self.power_curve(power, windspeed, bins_p, bins_w, std_p, std_w, self.bin_num)
            self.histograms[file_id] = pc.flatten()
            
    def dataset_js(self, model):
        model.train()
        js_loss = []
        # reset js_ratio
        self.js_ratio = 1.
        for _, (turbine_data, file_id) in enumerate(self.datasets):
            # turbine_data = self.turbinedata[i]
            turbine_data = turbine_data[~torch.isnan(turbine_data).any(axis=1)]
            data_input, _ = time_ahead(turbine_data, training=False)
            pc = self.histograms[file_id]
            
            pc = pc.to(DEVICE)
            data_input = data_input.to(DEVICE)
            # make prediction
            power_predict, _ = model(data_input.reshape(1, len(data_input), -1), training=False)
            windspeed = data_input[:, [1]]
            
            # data clip
            power_predict = power_predict[OFFSET-TIME_AHEAD:, :]
            windspeed = windspeed[OFFSET-TIME_AHEAD:, :]
            
            pc0 = self.power_curve(
                power_predict, 
                windspeed, 
                self.bins[file_id][0], 
                self.bins[file_id][1], 
                self.bins[file_id][2],
                self.bins[file_id][3],
                self.bin_num)
            pc0 = pc0.flatten()
            
            js_i = js_div(pc0, pc)
            pc_record.append([pc0.cpu(), pc.cpu()])
            # debug
            # js_record['pc'].append([power_predict, windspeed, pc0, pc, js_i])
                                
            js_loss.append(js_i)
            
        js_loss = sum(js_loss)/len(self.datasets)
        

        return js_loss * self.js_ratio
    
        
    def data_checker(self, data):
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        return data
        
    def power_curve(self, power, windspeed, bins_p, bins_w, std_p, std_w, bins, case=1):
        power = self.data_checker(power)
        windspeed  =self.data_checker(windspeed)
        power = power.unsqueeze(0)
        windspeed = windspeed.unsqueeze(0)
        
        bins_p = bins_p.to(power.device)
        bins_w = bins_w.to(windspeed.device)
        
        if case == 0:
            # scale output
            power = power - power.mean()
            power = power * (bins_p.max()/power.max())
        elif case == 1:
            # norm ouput
            power = (power -power.mean()) / power.std()  
        
        brandwidth_sigma = torch.pow(torch.tensor(bins, dtype=torch.float32), -1/5)*0.05
        # dynamic brandwidth
        if power.std() < 1e-2 or power.mean() < bins_p.min() or power.mean() > bins_p.max():
            self.js_ratio = 0
        # else:
        #     self.js_ratio = 1.
        brandwidth_p = power.std()*brandwidth_sigma
        brandwidth_w = std_w*brandwidth_sigma
            
        _, kernel_p = marginal_pdf(power, bins_p, brandwidth_p, 1e-10)
        _, kernel_w = marginal_pdf(windspeed, bins_w, brandwidth_w, 1e-10)
        pc = joint_pdf(kernel_p, kernel_w)
        return pc
    
    
def kl_div(p, q):
    kl = F.kl_div(p.log(), q, reduction='sum')
    return kl


def js_div(p: Tensor, q: Tensor) -> Tensor:
    m = 0.5 * (p + q)
    js = 0.5 * (kl_div(p, m) + kl_div(q, m))
    return js

  
def data_sampler(df, window_size, step):
    i = 0
    while i+window_size - 1 < len(df):
        yield df[i:i+window_size]
        i += step


def add_noise(data, e):
    # logger.info(f'add noise with ratio of {e}')
    noise = np.random.randn(len(data), 1)
    return data*(1+e*noise)


def feature_spliter(data):
    input_index = [0, 2, 4]
    target_index = [3]
    return data[:, input_index], data[:, target_index]


def time_ahead(data, noise_rate = 0, training=True):
    input_data, target_data = feature_spliter(data)
    wind_speed = input_data[TIME_AHEAD:, [1]]
    wind_speed_predict = input_data[TIME_AHEAD:, [2]]
    
    if training:
        # logger.info('time_ahead, training mode')
        historical_wind_speed = wind_speed[:OFFSET-TIME_AHEAD]
        # future_wind_speed = wind_speed[-TIME_AHEAD:]
        future_wind_speed = wind_speed_predict[-TIME_AHEAD:]
        if noise_rate > 0:
            print('running')
            # logger.info('add training noise')
            # noise = torch.randn(TIME_AHEAD, 1)
            # add noise normal distribution with mean and deviration
            noise = torch.normal(0, 1, (TIME_AHEAD, 1))
            future_wind_speed = future_wind_speed + noise * noise_rate
        # uncomment this line to use predicted wind speed data
        # wind_speed = torch.cat([historical_wind_speed, future_wind_speed])
        
    input_data = torch.cat(
        [input_data[TIME_AHEAD:, [0]],
        wind_speed,
        target_data[:-TIME_AHEAD, [0]]],
        dim=1)
    
    target_data = target_data[OFFSET:, :]
    return input_data, target_data


class NanSegDataset(torch.utils.data.Dataset):
    window_size = WINDOW_SIZE
    dtype = torch.float32
    
    def __init__(self, root_dir, start_date=None, noise_rate=None, test_len=2):
        self.root_dir = root_dir
        self.start_date = start_date
        self.noise_rate = noise_rate
        self.filenames = get_file_list(root_dir, r'.*csv')
        self.end_date = self.__get_end_date(start_date, test_len)
        print(f'testset: {self.start_date} {self.end_date}')
        self.input_ = []
        self.target_ = []
        self.scalers = {}
        self.__init_dataset()
        
    def __get_end_date(self, start_date, test_len):
        start_date = datetime.date(*[int(i) for i in start_date.split('-')])
        if start_date.month + test_len == 13:
            return '2020-12-15'
        else:
            end_date = datetime.date(start_date.year, start_date.month+test_len, 1)
            return end_date.strftime('%Y-%m-%d')
        
    def __init_dataset(self):
        self.testset = []
        for data, item in zip(DATASET, self.filenames):
            full_data = []
            # delete data after 2020-12-15
            mask = (data.index > self.start_date) & (data.index < self.end_date)
            data = data.loc[mask]
            data = data_clean(data)
            mean = torch.as_tensor(data.mean(), dtype=self.dtype)
            std = torch.as_tensor(data.std(), dtype=self.dtype)
            
            data = (data - mean) / std
            
            # store mean and std of turbine data
            self.scalers[item] = [mean, std]
            # add noise to wind speed before make segmentation
            
            wind_speed = data.iloc[:, [2]]
            # noise = np.random.randn(len(data), 1)
            # add noise normal distribution with mean and deviration
            noise = np.random.normal(0, 1, (len(data), 1))
            wind_speed_with_noise = wind_speed + noise * self.noise_rate
            wind_speed_predict = data.iloc[:, [4]]
            # add simulated wind speed data
            # data.iloc[:, [2]] = wind_speed_predict.values
            data.iloc[:, [2]] = wind_speed_with_noise
            
            # split the test data to avoid nan
            for data_seg in nan_seg(data):
                if len(data_seg) > self.window_size:
                    data_seg = torch.as_tensor(data_seg.values, dtype=self.dtype)
                    data_in, data_target = time_ahead(data_seg, training=False)
                    full_data.append([data_in, data_target])
            self.testset.append(full_data)
            
    def set_noise_rate(self, value):
        self.noise_rate = value
        self.__init_dataset()

    def __getitem__(self, index): 
        return self.testset[index], self.filenames[index]
         
    def __len__(self):
        return len(self.filenames)
        
        
class TestDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, start_date=None, noise_rate=None):
        self.root_dir = root_dir
        self.start_date = start_date
        self.noise_rate = noise_rate
        self.filenames = get_file_list(root_dir, r'.*csv')
        self.input_ = []
        self.target_ = []
        self.scalers = {}
        self.__init_dataset()
        
    def __init_dataset(self):
        for item in self.filenames:
            data = read_data(os.path.join(self.root_dir, item))
            # delete data after 2020-12-15
            data = data.loc[self.start_date:'2020-12-15']
            data = data_clean(data)
            # data.dropna(inplace=True)
            # delete nan value in test dataset
            data = data[~np.isnan(data).any(axis=1)]
            scaler = preprocessing.StandardScaler()
            scaler = scaler.fit(data)
            data = scaler.transform(data)
            data = torch.as_tensor(data, dtype=torch.float32)
            
            data_in, data_target = time_ahead(data, training=False)

            self.scalers[item] = scaler
            self.input_.append(data_in)
            self.target_.append(data_target)
            
    def set_noise_rate(self, value):
        self.noise_rate = value

    def __getitem__(self, index):
        data_in = self.input_[index]
        target = self.target_[index]
        
        if self.noise_rate and self.noise_rate>0:
            wind_speed = data_in[:, [1]]
            noise = torch.randn(len(data_in), 1)
            wind_speed = wind_speed*(1+self.noise_rate*noise)
            data_in = torch.cat(
                [data_in[:, [0]],
                 wind_speed,
                 data_in[:, [2]]], dim=1)
            
        return data_in, target, self.filenames[index]
        
    def __len__(self):
        return len(self.filenames)


class SeqDataset_KFold(torch.utils.data.Dataset):
    window_size = WINDOW_SIZE
    step = STEP
    dtype = torch.float32
    
    def __init__(self, root_dir, transform=None, file_list=None,
                 split_date=None, train_len=None, noise_rate=None):
        self.root_dir = root_dir
        self.file_list = file_list
        self.files = [os.path.join(self.root_dir, item) for item in self.file_list]
        self.flag, (self.start, self.end) = self.date_split(split_date, train_len=train_len)
        print(f'training dataset flag: {self.flag}, {self.start}, {self.end}, {train_len}')
        self.noise_rate = noise_rate
        self.__init_dataset()
    
    def __init_dataset(self):
        self.samples = []
        self.mean, self.std, self.whole_data = self.__dataset_normalizeation()
        for data in self.__data_loader():
            for data_seg in nan_seg(data):
                if len(data_seg) > self.window_size:
                    for sample in data_sampler(data_seg, self.window_size, self.step):
                        
                        sample = torch.as_tensor(sample.values, dtype=self.dtype)
                        
                        sample.sub_(self.mean).div_(self.std)
                        # inday
                        # self.samples.append(feature_spliter(sample))
                        # day ahead
                        self.samples.append(time_ahead(sample, noise_rate=self.noise_rate))
    
    def __dataset_normalizeation(self):
        
        whole_data = []
        for data in self.__data_loader():
            
            whole_data.append(data)
            
        whole_data = pd.concat(whole_data, axis=0, ignore_index=True)
        
        mean = torch.as_tensor(whole_data.mean(), dtype=self.dtype)
        std = torch.as_tensor(whole_data.std(), dtype=self.dtype)
        
        return mean, std, whole_data
    
    def date_split(self, split_date, train_len=10):
        split_date = datetime.date(*[int(i) for i in split_date.split('-')])
        months = list(range(1, 13))
        if split_date.month - train_len >= 1:
            # means that the train data does not apart
            node = datetime.date(split_date.year, split_date.month-train_len, 1)
            return 0, (node.strftime('%Y-%m-%d'), split_date.strftime('%Y-%m-%d'))
        else:
            # means that the test and no use data dose not apart
            node = datetime.date(split_date.year, months[split_date.month - train_len-1], 1)
            return 1, (split_date.strftime('%Y-%m-%d'), node.strftime('%Y-%m-%d'))
    
    def read_data_segment(self, data):
        mask = (data.index > self.start) & (data.index < self.end)
        if self.flag:
            # the training data would be seprate by nan values
            data.loc[mask] = np.NAN
            return data
        else:
            return data.loc[mask]
                
    def __data_loader(self):
        for data in DATASET:
        # for data_file in self.file_list:
        #     file_path = os.path.join(self.root_dir, data_file)
        #     data = read_data(file_path)
            data = data_clean(data)
            data = data.loc[:'2020-12-15']
            data = self.read_data_segment(data)
            yield data
            
    def datasets(self, norm=False, raw=True):
        # for data in self.__data_loader():
        for data, data_file in zip(DATASET, self.file_list):
        # for data_file in self.file_list:
        #     file_path = os.path.join(self.root_dir, data_file)
        #     data = read_data(file_path) 
            data = data_clean(data)
            # data = data.loc[self.start_date:SPLIT_DATE]
            data = self.read_data_segment(data)
            if raw:
                # keep the nan value
                data = torch.as_tensor(data.values, dtype=self.dtype)
                if norm:
                    yield data.sub_(self.mean).div(self.std), data_file
                else:
                    yield data, data_file
            else:
                # remove the nan value
                data = data[~np.isnan(data).any(axis=1)]
                data = torch.as_tensor(data.values, dtype=self.dtype)
                if norm:
                    yield data.sub_(self.mean).div_(self.std), data_file
                else:
                    yield data, data_file

    def __getitem__(self, index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)
     

class SeqDataset(torch.utils.data.Dataset):
    window_size = WINDOW_SIZE
    step = STEP
    dtype = torch.float32
    
    def __init__(self, root_dir, transform=None, file_list=None, start_date=None, noise_rate=None):
        self.root_dir = root_dir
        self.file_list = file_list
        self.start_date = start_date
        self.noise_rate = noise_rate
        self.__init_dataset()
    
    def __init_dataset(self):
        self.samples = []
        self.mean, self.std, self.whole_data = self.__dataset_normalizeation()
        for data in self.__data_loader():
            for data_seg in nan_seg(data):
                if len(data_seg) > self.window_size:
                    for sample in data_sampler(data_seg, self.window_size, self.step):
                        
                        sample = torch.as_tensor(sample.values, dtype=self.dtype)
                        
                        sample.sub_(self.mean).div_(self.std)
                        # inday
                        # self.samples.append(feature_spliter(sample))
                        # day ahead
                        self.samples.append(time_ahead(sample, noise_rate=self.noise_rate))
    
    def __dataset_normalizeation(self):
        
        whole_data = []
        for data in self.__data_loader():
            
            whole_data.append(data)
            
        whole_data = pd.concat(whole_data, axis=0, ignore_index=True)
        
        mean = torch.as_tensor(whole_data.mean(), dtype=self.dtype)
        std = torch.as_tensor(whole_data.std(), dtype=self.dtype)
        
        return mean, std, whole_data
    
    def __data_loader(self):
        for data_file in self.file_list:
            file_path = os.path.join(self.root_dir, data_file)
            data = read_data(file_path)
            data = data_clean(data)
            yield data.loc[self.start_date:SPLIT_DATE]
                
    def datasets(self, norm=True, raw=True):
        # for data in self.__data_loader():
        for data_file in self.file_list:
            file_path = os.path.join(self.root_dir, data_file)
            data = read_data(file_path) 
            data = data_clean(data)
            data = data.loc[self.start_date:SPLIT_DATE]
            if raw:
                # keep the nan value
                data = torch.as_tensor(data.values, dtype=self.dtype)
                if norm:
                    yield data.sub_(self.mean).div(self.std), data_file
                else:
                    yield data, data_file
            else:
                # remove the nan value
                data = data[~np.isnan(data).any(axis=1)]
                data = torch.as_tensor(data.values, dtype=self.dtype)
                if norm:
                    yield data.sub_(self.mean).div_(self.std), data_file
                else:
                    yield data, data_file

    def __getitem__(self, index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)


class Record():
    
    def __init__(self):
        self.__record = {}
    
    def update(self, name, value):
        if name in self.__record.keys():
            self.__record[name].append(value)
        else: 
            self.__record[name] = []
            self.__record[name].append(value)
            
    def data(self, name):
        if name in self.__record.keys():
            return self.__record[name]
        else:
            return []
    
    def names(self):
        return self.__record.keys()


def save_checkpoint(state, filename):
    torch.save(state, filename)

   
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
                
                
                output, _ = model(input_)
                
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
    return predict


def plot(predict, target):
    plt.plot(predict.cpu().flatten(), color='black')
    plt.plot(target.cpu().flatten())
    plt.savefig('result.png')
    plt.show()


def predicter(model_predict, data):
    result = []
    train_len = OFFSET
    i = 0
    while i + train_len < len(data) - TIME_AHEAD:
        pred = evaluater(model_predict, data[i:i+train_len, :])
        result.append(pred)
        i += TIME_AHEAD
    return torch.cat(result, dim=0)



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


def load_model(experiment_id, epochs):
    with open(f'record/{experiment_id}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['checkpoint'][epochs-1]

def backup_record(experiment_id):
    src = f'record/{experiment_id}.pkl'
    dest = f'record/{experiment_id}.bc.pkl'
    shutil.copyfile(src, dest)


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

        
def tester(model, criterion, testdataset, logger):
    # switch to evaluate mode
    model.eval()
    
    record = {}
    
    for i, (test_input, test_target, name) in enumerate(testdataset):
        test_input = test_input.to(DEVICE)
        test_target = test_target.to(DEVICE)
        predict = evaluater(model, test_input)
        predict = predict[OFFSET-TIME_AHEAD:, :]

        test_loss = criterion(predict, test_target)
        
        logger.info(f'{name}: {test_loss.item()}')
        
        test_record = {
            'index': i,
            'pred': predict,
            'targ': test_target,
            'loss': test_loss.item(),
        }
        
        record[name] = test_record
        
    return record


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0)
    elif type(layer) == nn.LSTM:
        for name, param in layer.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
                        
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
    
    awl = AutomaticWeightedLoss(2)
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

    # model.apply(init_weights)    
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.Adam([
    #                 {'params': model.parameters()},
    #                 {'params': awl.parameters(), 'weight_decay': 0}
    #             ], lr=lr)
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
            # evaluation
            test_record = test_noise_rate(testdataset, model, criterion, test_noise, logger)
            
            # save checkpoint
            record['checkpoint'][epoch] = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
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
    parser.add_argument('--id', type=str, default='99999', metavar='N',
                        help='experiment id (default: 0)')
    parser.add_argument('--start', type=str, default='2020-11-01', metavar='XXXX-XX-XX',
                        help='start date (default: 2020-11-01)')
    parser.add_argument('--train_len', type=int, default=10, metavar='N',
                        help='training date legenth')
    parser.add_argument('--js_mode', action='store_true', default=False,
                        help='enable js loss')
    parser.add_argument('--noise_rate', type=float, default=0, metavar='V',
                        help='noise for target')
    parser.add_argument('--test_noise', action='store_true', default=True,
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
    
    # os.chdir(r'D:\DT')
    # os.chdir(r'C:\Users\Administrator\Desktop\DT')
    os.chdir(r'D:\gjx\cyt\风能\DT')
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