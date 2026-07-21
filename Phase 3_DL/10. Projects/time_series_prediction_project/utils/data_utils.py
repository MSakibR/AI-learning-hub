import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_csv(path, column_name):
    """Load a CSV and return the selected column as numpy array"""
    df = pd.read_csv(path)
    return df[column_name].values.reshape(-1, 1)


def create_sequences(data, seq_length=10):
    """Convert 1D array into sequences for RNN/LSTM/GRU"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def normalize_data(train, test):
    """Normalize train and test data"""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled, scaler
