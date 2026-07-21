import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from rnn_audio_model import build_audio_model
from utils.audio_utils import load_audio, extract_mfcc

# Load data
data_dir = "data/audio"
emotions = ["happy", "sad", "angry", "neutral"]
X, y = [], []

for idx, emotion in enumerate(emotions):
    folder = os.path.join(data_dir, emotion)
    for file in os.listdir(folder):
        audio = load_audio(os.path.join(folder, file))
        mfcc = extract_mfcc(audio)  # shape (100, 40)
        X.append(mfcc)
        y.append(idx)

X = np.array(X)
y = to_categorical(np.array(y), num_classes=len(emotions))

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = build_audio_model("LSTM", input_shape=(100, 40), num_classes=len(emotions))
model.summary()

# Train
history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32
)

# Save model
model.save("models/audio_emotion_lstm.h5")
