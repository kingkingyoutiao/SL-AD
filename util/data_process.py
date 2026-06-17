import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
    def remove_outliers(self, data, method=1, axis=0):
        if method not in [1, 2]:
            raise ValueError("Invalid method for outlier removal. Choose either 1 or 2.")

        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)

        if method == 1:
            lower_bound = mean - 2 * std
            upper_bound = mean + 2 * std
        elif method == 2:
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
        # 替换异常值为NaN
        data[data < lower_bound] = np.nan
        data[data > upper_bound] = np.nan
        return data
    def linear_interpolation(self, data, axis=0):
        if axis == 0:
            for i in range(data.shape[1]):
                if np.isnan(data[0, i]):
                    valid_values = data[1:, i][~np.isnan(data[1:, i])]
                    if len(valid_values) > 0:
                        data[0, i] = np.mean(valid_values)
                if np.isnan(data[-1, i]):
                    valid_values = data[:-1, i][~np.isnan(data[:-1, i])]
                    if len(valid_values) > 0:
                        data[-1, i] = np.mean(valid_values)

            data = pd.DataFrame(data).interpolate(axis=axis).values
        elif axis == 1:
            for i in range(data.shape[0]):
                if np.isnan(data[i, 0]):
                    valid_values = data[i, 1:][~np.isnan(data[i, 1:])]
                    if len(valid_values) > 0:
                        data[i, 0] = np.mean(valid_values)
                if np.isnan(data[i, -1]):
                    valid_values = data[i, :-1][~np.isnan(data[i, :-1])]
                    if len(valid_values) > 0:
                        data[i, -1] = np.mean(valid_values)

            data = pd.DataFrame(data).interpolate(axis=axis).values
        return data
    def normalize(self, data, method=1, axis=0):
        if method == 1:
            scaler = StandardScaler()
        elif method == 2:
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid method for normalization. Choose either 1 or 2.")

        if axis == 0:
            data = scaler.fit_transform(data)
        elif axis == 1:
            data = scaler.fit_transform(data.T).T
        return data
    def process(self, data, operations, methods, axis):
        valid_operations = ['A', 'C', 'M']
        methods={o:m for o,m in zip(operations,methods)}
        for op in operations:
            if op not in valid_operations:
                raise ValueError(f"Invalid operation '{op}'. Valid operations are {valid_operations}")

        if 'A' in operations:
            if methods['A'] == 0:
                data = self.remove_outliers(data, method=1, axis=0 if axis == 'column' else 1)
            elif methods['A'] == 1:
                data = self.remove_outliers(data, method=2, axis=0 if axis == 'column' else 1)
        if 'C' in operations:
            data = self.linear_interpolation(data, axis=0 if axis == 'column' else 1)
        if 'M' in operations:
            if methods['M'] == 0:
                data = self.normalize(data, method=1, axis=0 if axis == 'column' else 1)
            elif methods['M'] == 1:
                data = self.normalize(data, method=2, axis=0 if axis == 'column' else 1)
        return data
def scaler(a):
    return (a-np.mean(a))/np.std(a)