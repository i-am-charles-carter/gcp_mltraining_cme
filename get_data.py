import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight


def min_max(data, scale_min=-1, scale_max=1):
    f_max = np.max(data)
    f_min = np.min(data)
    denominator = f_max - f_min
    if denominator == 0:
        scaled_data = np.ones_like(data) * scale_min
    else:
        scaled_data = ((data - f_min) / denominator) * (scale_max - scale_min) + scale_min
    return scaled_data


def calculate_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, class_weights))
    return class_weights


def prepare_data(data, x_window_size, y_window_size):
    num_days, ts = data.shape
    num_samples = (ts - x_window_size - y_window_size + 1) * num_days
    X = np.empty((num_samples, x_window_size))
    Y = np.empty(num_samples, dtype=object)  # specify dtype as object for storing string labels
    sample_index = 0
    bin_labels = {1: 'Very Bearish', 2: 'Bearish', 3: 'Flat', 4: 'Bullish', 5: 'Very Bullish'}
    for i in range(num_days):
        x_windows = np.lib.stride_tricks.sliding_window_view(
            data[i, :ts - y_window_size], (x_window_size,))
        y_windows = np.lib.stride_tricks.sliding_window_view(
            data[i, x_window_size:], (y_window_size,))
        num_windows = x_windows.shape[0]
        X[sample_index:sample_index + num_windows] = x_windows
        Y[sample_index:sample_index + num_windows] = [bin_labels[i] 
                                                      for i in np.digitize(np.diff(y_windows, axis=1).sum(axis=1), 
                                                                           [-np.inf, -3, -1, 1, 3, np.inf])]
        sample_index += num_windows
    class_weights = calculate_class_weights(Y)
    return X, Y, class_weights


def prepare_and_split_data(data, x_window_size, y_window_size, test_size=0.2):
    num_days, ts = data.shape
    X, Y, class_weights = prepare_data(data, x_window_size, y_window_size)
    indices = np.random.permutation(len(X))
    X = X[indices]
    Y = Y[indices]
    # Split into train and test sets
    test_set_size = int(len(X) * test_size)
    X_train = X[test_set_size:]
    X_test = X[:test_set_size]
    Y_train = Y[test_set_size:]
    Y_test = Y[:test_set_size]
    return X_train, X_test, Y_train, Y_test, class_weights


def get_data(product, x_window_size, y_window_size, test_size=0.2):
    # Load and preprocess data
    data = pd.read_feather(r"C:\Users\charl\vscode_projects\cme_classify\gcp_mltraining_cme\SPX500_USD_2018_2022_hf_14_21.feather")
    data = data.set_index("datetime")
    data = data[f'close']
    # Reformat the data so the price is in 0.25 increments
    data = round(data * 4) / 4
    days = len(data.groupby(data.index.date))
    data = data.to_numpy().reshape(days, -1)
    # Split into train and test sets
    train_X, test_X, train_y, test_y, class_weights = prepare_and_split_data(data, x_window_size, y_window_size, test_size=test_size)
    ds = train_X, test_X, train_y, test_y, class_weights
    return ds
