import re
import pandas as pd
from torch.utils.data import DataLoader
import torch
import os
import datetime
import numpy as np
from configs import *
from warnings import simplefilter
simplefilter(action="ignore",category=FutureWarning)


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

ROOT_DIR = '江苏十分钟数据'
FILE_LIST = get_file_list(ROOT_DIR, r'.*csv')
# print(FILE_LIST)
FILE_LIST = ['江苏十分钟数据/' + f for f in FILE_LIST]
if not os.path.exists('record'):
    os.mkdir('record')

DATASET = make_dataset(FILE_LIST)

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
            end_date = datetime.date(start_date.year, start_date.month + test_len, 1)
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
            node = datetime.date(split_date.year, split_date.month - train_len, 1)
            return 0, (node.strftime('%Y-%m-%d'), split_date.strftime('%Y-%m-%d'))
        else:
            # means that the test and no use data dose not apart
            node = datetime.date(split_date.year, months[split_date.month - train_len - 1], 1)
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


def data_sampler(df, window_size, step):
    i = 0
    while i + window_size - 1 < len(df):
        yield df[i:i + window_size]
        i += step


def feature_spliter(data):
    input_index = [0, 2, 4]
    target_index = [3]
    return data[:, input_index], data[:, target_index]


def time_ahead(data, noise_rate=0, training=True):
    input_data, target_data = feature_spliter(data)
    wind_speed = input_data[TIME_AHEAD:, [1]]
    wind_speed_predict = input_data[TIME_AHEAD:, [2]]

    if training:
        # logger.info('time_ahead, training mode')
        historical_wind_speed = wind_speed[:OFFSET - TIME_AHEAD]
        future_wind_speed = wind_speed[-TIME_AHEAD:]
        # future_wind_speed = wind_speed_predict[-TIME_AHEAD:]
        if noise_rate > 0:
            print('running')
            # logger.info('add training noise')
            # noise = torch.randn(TIME_AHEAD, 1)
            # add noise normal distribution with mean and deviration
            noise = torch.normal(0, 1, (TIME_AHEAD, 1))
            future_wind_speed = future_wind_speed + noise * noise_rate
        # uncomment this line to use predicted wind speed data
        wind_speed = torch.cat([historical_wind_speed, future_wind_speed])

    input_data = torch.cat(
        [input_data[TIME_AHEAD:, [0]],
         wind_speed,
         # wind_speed_predict,
         # input_data[:-TIME_AHEAD, [1]],
         target_data[:-TIME_AHEAD, [0]]],
        dim=1)

    target_data = target_data[OFFSET:, :]
    return input_data, target_data