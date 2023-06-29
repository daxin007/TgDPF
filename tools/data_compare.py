import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import pickle


def read_data(filepath, freq='15min', index='日期'):
    df = pd.read_csv(filepath, encoding='gbk')
    df[index] = pd.to_datetime(df[index])
    df.set_index([index], inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    full_perid = pd.date_range(start='20200101', end='20210101',freq=freq, closed='left')
    df = pd.DataFrame(df, index=full_perid)
    return df

def mergefile(file1, file2):
    
    df1 = read_data(file1)
    df2 = read_data(file2)
    
    df = pd.concat([df1, df2], axis=1)
    df["wind_speed"] = np.sqrt(df.U**2 + df.V**2)
    df_resampled = df.resample('10min').mean()
    df_resampled = df_resampled.interpolate(limit=1)
    # plt.plot(df.wind_speed, marker="o")
    # plt.plot(df_resampled.wind_speed, marker="o")
    # plt.show()
    df_utc = df_resampled.index.tz_localize('UTC')
    df_resampled.index = df_utc.tz_convert('Asia/Shanghai')
    full_perid = pd.date_range(start='20200101', end='20210101',freq='10min', closed='left')
    df = pd.DataFrame(df, index=full_perid)
    return df_resampled


def run(task, name):
    df1 = mergefile(*task)
    df1.index = df1.index.tz_localize(None)
    df1.to_csv(f'synthetic_wind_speed/{name}.csv')
    with open(f'synthetic_wind_speed/{name}.pkl', 'wb') as f:
        pickle.dump(df1, f)
    for i in range(1, 26):
        df2 = read_data(f'江苏十分钟数据/D{i}.csv', freq='10min', index='时间')
        compare = df2['平均风速'][6*24:]
        df2 = df2[:-6*24]
        df2['compare'] = compare.values
        df2 = df2[~np.isnan(df2).any(axis=1)]
        y1 = df2['平均风速']
        y2 = df2['compare']
        y1_ = (y1 - y1.mean())/y1.std()
        y2_ = (y2 - y2.mean())/y2.std()
        print(f'D{i},{r2_score(y1, y2)},{mean_squared_error(y1_, y2_)},{np.corrcoef(y1_, y2_)[0][1]}')
        
        df1[f'D{i}'] = df2['平均风速']
    # df1['mse'] = (df1.wind_speed_real - df1.wind_speed)
    
    # df1.to_excel('tmp.xls')
    df1 = df1[~np.isnan(df1).any(axis=1)]
    
    y_pred = df1.wind_speed.to_numpy()
    
    y_pred_ = (y_pred - y_pred.mean())/y_pred.std()
    print('name,r2,mse,corr')
    for i in range(1, 26):
        
        y_true = df1[f'D{i}'].to_numpy()
        y_true_ = (y_true - y_true.mean())/y_true.std()
     
        print(f'D{i},{r2_score(y_true, y_pred)},{mean_squared_error(y_true_, y_pred_)},{np.corrcoef(y_true_, y_pred_)[0][1]}')
        # np.sum((y_true_ - y_pred_)**2)/len(y_true_))
        plt.plot(y_true_, 's', markersize=1, alpha=0.8)
        # plt.scatter(y_true, y_pred, s=1, alpha=0.1)
        
    plt.plot(y_pred_,'s', markersize=1, color='red', label='predict')
    # plt.plot(df1.D1, label='real')
    plt.legend()
    # plt.scatter(df1.wind_speed, df1.wind_speed_real)
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    task1 = ('synthetic_wind_speed/绔欑偣1_+120.080_+032.212_U_1000hpa.csv',
             'synthetic_wind_speed/绔欑偣1_+120.080_+032.212_V_1000hpa.csv')
    task2 = ('synthetic_wind_speed/绔欑偣2_+121.966_+034.611_U_1000hpa.csv',
             'synthetic_wind_speed/绔欑偣2_+121.966_+034.611_V_1000hpa.csv')
    name = 'task2'
    run(eval(name), name)