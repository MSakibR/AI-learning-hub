# folder structure with all directories and files:

emotion_detection_audio_project/
│
├── data/
│   ├── audio/                  # Audio files organized by emotion
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── angry/
│   │   └── neutral/
│   └── test_audio/             # Audio files for testing predictions
│
├── models/                     # Trained models will be saved here
│
├── utils/
│   ├── audio_utils.py          # Load audio, extract features (MFCC)
│   └── visualization.py        # Plotting training metrics or confusion matrix
│
├── rnn_audio_model.py          # RNN/LSTM/GRU model for emotion detection
├── train.py                    # Training script
├── predict.py                  # Predict emotions for new audio
├── requirements.txt            # Packages
└── README.md                   # Project documentation


# pip install -r requirements.txt


# Powershell
New-Item -ItemType Directory -Path "emotion_detection_audio_project\data\audio\happy","emotion_detection_audio_project\data\audio\sad","emotion_detection_audio_project\data\audio\angry","emotion_detection_audio_project\data\audio\neutral","emotion_detection_audio_project\data\test_audio","emotion_detection_audio_project\models","emotion_detection_audio_project\utils"; New-Item -ItemType File -Path "emotion_detection_audio_project\utils\audio_utils.py","emotion_detection_audio_project\utils\visualization.py","emotion_detection_audio_project\rnn_audio_model.py","emotion_detection_audio_project\train.py","emotion_detection_audio_project\predict.py","emotion_detection_audio_project\requirements.txt","emotion_detection_audio_project\README.md"



# Key Notes:

- MFCC Features: Converts audio waveform into features suitable for RNN/LSTM/GRU.
- Model Flexibility: Switch between RNN, LSTM, GRU easily in rnn_audio_model.py.
- Dropout: Helps prevent overfitting for small audio datasets.
- Visualization: Plot training metrics and confusion matrix to see performance.