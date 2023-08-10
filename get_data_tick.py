import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
import plotly.graph_objects as go
from tick_to_ohlcv import get_ohlcv
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import plotly.express as px


def plot_hist(df):
    for i in range(1, 6):
        df[f'shifted_{i}'] = df['close'].shift(i)

    # Calculate the differences
    df_diff = df.iloc[:, 1:].diff(axis=1)
    # Sum the rows
    df_diff['sum'] = df_diff.sum(axis=1)
    # Define the bin edges
    # bin_edges = [-np.inf, -0.5, 0, 0.5, np.inf]
    bin_edges = [-np.inf, 0.5, np.inf]
    # Calculate the counts for each bin
    counts, _ = np.histogram(df_diff['sum'].dropna(), bins=bin_edges)
    # Calculate the percentages
    percentages = counts / counts.sum() * 100
    # Create a DataFrame from the bin edges and counts
    df_hist = pd.DataFrame({
        'bins': ['(-inf, 0.5)', '[0.5, inf)'],
        'counts': counts,
        'percentage': percentages
    })
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(x=df_hist['bins'], y=df_hist['counts'], text=df_hist['percentage'])
    ])
    # Customize aspect
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.show()


def calculate_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, class_weights))
    return class_weights


def prepare_day_data(day_data, n_timesteps_in, n_timesteps_out):
    day_data_array = day_data['close'].values

    # Create rolling windows for X
    X = np.lib.stride_tricks.sliding_window_view(day_data_array, (n_timesteps_in,))

    # Create rolling windows for Y, shifted by n_timesteps_in
    Y = np.lib.stride_tricks.sliding_window_view(day_data_array[n_timesteps_in:], (n_timesteps_out,))

    # Drop the last n_timesteps_out rows from X and the first row from Y
    X = X[:-n_timesteps_out]

    # Transform X by applying Min-Max scaling row-wise
    X = (X - X.min(axis=1).reshape(-1, 1)) / (X.max(axis=1) - X.min(axis=1)).reshape(-1, 1)

    # Transform Y by calculating the difference, summing it, and then binning and labeling it
    Y_diff_sum = np.diff(Y, axis=1).sum(axis=1)
    bin_labels = {1: 'Very Bearish', 2: 'Bearish', 3: 'Bullish', 4: 'Very Bullish'}
    bin_edges = [-np.inf, -0.5, 0, 0.5, np.inf]
    # bin_edges = [-np.inf, 0.5, np.inf]
    # Find the indices where Y_diff_sum belongs in the bins
    bin_indices = np.digitize(Y_diff_sum, bins=bin_edges)
    # Map bin indices to your custom bin labels
    # custom_bin_labels = {1: 'Very Bearish', 2: 'Bullish'}
    Y = np.array([bin_labels[i] for i in bin_indices])

    return X, Y


def prepare_data(data, n_timesteps_in, n_timesteps_out):
    # Group the data by date
    grouped = data.groupby(data.index.date)

    # Loop through each day, prepare the data for that day, and append it to the lists
    X_list = []
    Y_list = []
    for _, group in grouped:
        X_day, Y_day = prepare_day_data(group, n_timesteps_in, n_timesteps_out)
        X_list.append(X_day)
        Y_list.append(Y_day)

    # Concatenate all the data together
    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)  # Assuming Y is a pandas Series

    return X, Y


def shuffle_and_split_data(X, Y, test_size=0.2, val_size=0.2):
    # Ensure that X and Y have the same length
    assert len(X) == len(Y), "X and Y must have the same length"

    # Shuffle the data
    # indices = np.random.permutation(len(X))
    # X = X[indices]
    # Y = Y[indices]
    # class_weights = calculate_class_weights(Y)
    # Split the data into training, testing, and validation datasets
    test_split_idx = int(len(X) * test_size)
    val_split_idx = int(len(X) * (test_size + val_size))
    X_test = X[:test_split_idx]
    Y_test = Y[:test_split_idx]
    X_val = X[test_split_idx:val_split_idx]
    Y_val = Y[test_split_idx:val_split_idx]
    X_train = X[val_split_idx:]
    Y_train = Y[val_split_idx:]
    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, Y_train)
    X_test_resampled, y_test_resampled = smote.fit_resample(X_test, Y_test)
    X_val_resampled, y_val_resampled = smote.fit_resample(X_val, Y_val)
    # Calculate the class counts after SMOTE
    unique_classes, class_counts_resampled = np.unique(y_train_resampled, return_counts=True)

    # Create a histogram using Plotly
    fig = px.bar(x=unique_classes, y=class_counts_resampled,
                 title="Class Distribution After SMOTE",
                 labels={'x': 'Class', 'y': 'Count'})

    fig.show()

    return (X_train_resampled, X_val_resampled, X_test_resampled, y_train_resampled, y_val_resampled, y_test_resampled,
            class_counts_resampled)




