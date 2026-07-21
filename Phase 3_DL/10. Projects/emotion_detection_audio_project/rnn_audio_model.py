from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout


def build_audio_model(
    model_type="LSTM", input_shape=(100, 40), num_classes=4, hidden_units=64
):
    """
    model_type: 'RNN', 'LSTM', 'GRU'
    input_shape: (timesteps, features)
    num_classes: number of emotion classes
    """
    model = Sequential()

    if model_type == "RNN":
        model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation="tanh"))
    elif model_type == "LSTM":
        model.add(LSTM(hidden_units, input_shape=input_shape, activation="tanh"))
    elif model_type == "GRU":
        model.add(GRU(hidden_units, input_shape=input_shape, activation="tanh"))
    else:
        raise ValueError("Choose RNN, LSTM, or GRU")

    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
