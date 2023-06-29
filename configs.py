import torch
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
torch.autograd.set_detect_anomaly(True)
SPLIT_DATE = '2020-11-01'

TIME_AHEAD = 6*24
WINDOW_SIZE = 5*6*24
STEP = 6*24
OFFSET = WINDOW_SIZE-TIME_AHEAD
Tensor = torch.tensor
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

else:
    DEVICE = torch.device('cpu')
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