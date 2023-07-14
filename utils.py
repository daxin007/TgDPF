import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F
import shutil
import pickle
from configs import *

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def marginal_pdf(values, bins, a, b=1e-10):
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

def kl_div(p, q):
    kl = F.kl_div(p.log(), q, reduction='sum')
    return kl


def js_div(p: Tensor, q: Tensor) -> Tensor:
    m = 0.5 * (p + q)
    js = 0.5 * (kl_div(p, m) + kl_div(q, m))
    return js


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

def load_model(experiment_id, epochs):
    with open(f'record/{experiment_id}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['checkpoint'][epochs-1]

def backup_record(experiment_id):
    src = f'record/{experiment_id}.pkl'
    dest = f'record/{experiment_id}.bc.pkl'
    shutil.copyfile(src, dest)