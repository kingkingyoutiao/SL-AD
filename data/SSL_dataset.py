import argparse
import pickle
import random
from tqdm.std import tqdm
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import List
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from util.timefeatures import time_features
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.week - 1) / 52.0 - 0.5

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:#
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)
def stamp2date(stamp):
    timeArray = time.localtime(stamp)
    date = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return date

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./datasets/magn_all', help='path of data')
parser.add_argument('--data_type', type=str, default='magn')
parser.add_argument('--cleaning', type=str, default='fill_0')
parser.add_argument('--filling', type=str, default='linear_interpolate')
parser.add_argument('--threshold_time', type=int, default=72)
parser.add_argument('--norm_data', type=str, default='oneSta_oneFea')
parser.add_argument('--norm_type', type=str, default='quartile_seg')
parser.add_argument('--fea_select', type=str, default='all')
parser.add_argument('--fea_use', type=str, default='Fourier_power_0_15')
parser.add_argument('--dataset_split_time', type=str, default='2022-01-01 00:00:00')
parser.add_argument('--input_length', type=str, default='7days')
parser.add_argument('--input_sel_type', type=str, default='Slide')
parser.add_argument('--input_window_size', type=int, default=1008)
parser.add_argument('--predict_size', type=int, default=7)
parser.add_argument('--class_type', type=str, default='binary_cls')
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--sample', type=str, default='undersampling')
parser.add_argument('--train_phase', type=str, default='train')

parser.add_argument('--epochs', type=int, default=4, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=0.00001, help='initial (base) learning rate')
parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='path for saving result models')
parser.add_argument('--results', type=str, default='./results', help='path for saving result models')
parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
parser.add_argument('--hidden_nc', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--optimizer', type=str, default='Adam', help='the optimizer: SGD|Adam')
parser.add_argument('--model_pre', type=str, default='Eq_Fore', help='network: Eq_Fore')
parser.add_argument('--model_cls', type=str, default='BiLSTM', help='network: MLP | BiLSTM')
parser.add_argument('--model_pred_state', type=str, default='resume')

parser.add_argument('--seq_len', type=int, default=1008, help='input sequence length of encoder')
parser.add_argument('--label_len', type=int, default=144*2, help='start token length of decoder')
parser.add_argument('--pred_len', type=int, default=1008, help='prediction sequence length')

parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
parser.add_argument('--c_out', type=int, default=3, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--seed', type=int, default=77, help='The random seed')
args1 = parser.parse_args()



