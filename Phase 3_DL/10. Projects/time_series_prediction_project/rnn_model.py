from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense


def build_rnn_model(model_type="LSTM", seq_length=10, features=1, hidden_units=50):
    """
    Build RNN/LSTM/GRU model
    model_type: 'RNN', 'LSTM', 'GRU'
    """
    model = Sequential()

    if model_type == "RNN":
        model.add(
            SimpleRNN(
                hidden_units, activation="tanh", input_shape=(seq_length, features)
            )
        )
    elif model_type == "LSTM":
        model.add(
            LSTM(hidden_units, activation="tanh", input_shape=(seq_length, features))
        )
    elif model_type == "GRU":
        model.add(
            GRU(hidden_units, activation="tanh", input_shape=(seq_length, features))
        )
    else:
        raise ValueError("Choose RNN, LSTM, or GRU")

    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
