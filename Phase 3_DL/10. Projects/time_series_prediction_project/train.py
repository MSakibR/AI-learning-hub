from utils.data_utils import load_csv, create_sequences, normalize_data
from rnn_model import build_rnn_model
import numpy as np

# Load Data
train_data = load_csv("data/train.csv", "Close")
test_data = load_csv("data/test.csv", "Close")

# Normalize
train_scaled, test_scaled, scaler = normalize_data(train_data, test_data)

# Create sequences
seq_length = 10
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

# Reshape for RNN input
X_train = X_train.reshape(-1, seq_length, 1)
X_test = X_test.reshape(-1, seq_length, 1)

# Build Model
model = build_rnn_model("LSTM", seq_length=seq_length, features=1, hidden_units=50)
model.summary()

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save Model
model.save("models/lstm_timeseries.h5")
