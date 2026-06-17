import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj=='type3':
        # lr_adjust = {m:n for m,n in zip(range)}
        lr_adjust = {
            130: 1e-6,
            260: 5e-7, 340: 1e-7, 400: 5e-8, 500: 1e-8
        }
    elif args.lradj=='type4':
        lr_adjust = {
            30: 1e-6,
            60: 5e-7, 80: 1e-7, 100: 5e-8, 120:1e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf',setting=None):
    """
    Results visualization
    """
    # print(true.shape, preds.shape)
    if setting[:3]=='CNN':
        plt.figure(figsize=(12,4))
        plt.plot(list(true[0:1008:6])+list(true[1008:]), label='GroundTruth', linewidth=2)
        if preds is not None:
            plt.plot(list(preds[0:1008:6])+list(preds[1008:]), label='Prediction', linewidth=2)
            plt.vlines([168], ymin=min(preds) - 0.2, ymax=max(preds) + 0.2, colors='red', linestyles='dashed',
                       label='Predict the commencement time')
        plt.legend()
        plt.savefig(name, bbox_inches='tight')
        return
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
        plt.vlines([len(true)],ymin=min(preds)-0.2,ymax=max(preds)+0.2,colors='red',linestyles='dashed',label='Predict the commencement time')
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def downsample_time_series(data, s=6, method='skip'):
    if len(data.shape) != 2:
        raise ValueError("输入数据必须是二维数组，形状为 (时间步数, 特征数)")
    n_time_steps, n_features = data.shape
    if method == 'skip':
        downsampled_data = data[::s]
    elif method == 'mean':
        downsampled_data = []
        for i in range(0, n_time_steps, s):
            window_data = data[i:i + s]
            window_mean = np.mean(window_data, axis=0)
            downsampled_data.append(window_mean)
        downsampled_data = np.array(downsampled_data)
    else:
        raise ValueError("无效的降采样方式。请选择 'skip' 或 'mean'。")
    return downsampled_data
