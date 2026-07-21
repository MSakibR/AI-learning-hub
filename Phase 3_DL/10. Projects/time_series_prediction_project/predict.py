from tensorflow.keras.models import load_model
from utils.data_utils import load_csv, create_sequences, normalize_data
from utils.visualization import plot_predictions
import numpy as np

# Load Model
model = load_model("models/lstm_timeseries.h5")

# Load Data
train_data = load_csv("data/train.csv", "Close")
test_data = load_csv("data/test.csv", "Close")

# Normalize
train_scaled, test_scaled, scaler = normalize_data(train_data, test_data)

# Create sequences
seq_length = 10
X_test, y_test = create_sequences(test_scaled, seq_length)
X_test = X_test.reshape(-1, seq_length, 1)

# Predict
predictions = model.predict(X_test)

# Inverse scale
predicted = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test)

# Plot
plot_predictions(actual, predicted, title="Time-Series Prediction")
