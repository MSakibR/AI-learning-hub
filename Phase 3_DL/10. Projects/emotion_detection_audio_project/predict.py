import numpy as np
from tensorflow.keras.models import load_model
from utils.audio_utils import load_audio, extract_mfcc

# Load model
model = load_model("models/audio_emotion_lstm.h5")
emotions = ["happy", "sad", "angry", "neutral"]

# Load new audio
audio_path = "data/test_audio/test1.wav"
audio = load_audio(audio_path)
mfcc = extract_mfcc(audio)
mfcc = np.expand_dims(mfcc, axis=0)  # shape (1, 100, 40)

# Predict
pred = model.predict(mfcc)
emotion_label = emotions[np.argmax(pred)]
confidence = np.max(pred)

print(f"Predicted Emotion: {emotion_label}, Confidence: {confidence:.2f}")
