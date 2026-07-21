# folder structure with all directories and files:

time_series_prediction_project/
│
├── data/
│   ├── train.csv              # Training data (time-series)
│   ├── test.csv               # Testing data
│   └── stock_prices.csv       # Example dataset
│
├── models/                    # Trained models will be saved here
│
├── utils/
│   ├── data_utils.py          # Functions to load, normalize, and create sequences
│   └── visualization.py       # Plotting functions
│
├── rnn_model.py               # RNN/LSTM/GRU model definitions
├── train.py                   # Training script
├── predict.py                 # Predict and visualize new data
├── requirements.txt           # Packages
└── README.md                  # Project documentation

# pip install -r requirements.txt

# Powershell
New-Item -ItemType Directory -Path "time_series_prediction_project\data","time_series_prediction_project\models","time_series_prediction_project\utils"; New-Item -ItemType File -Path "time_series_prediction_project\data\train.csv","time_series_prediction_project\data\test.csv","time_series_prediction_project\data\stock_prices.csv","time_series_prediction_project\utils\data_utils.py","time_series_prediction_project\utils\visualization.py","time_series_prediction_project\rnn_model.py","time_series_prediction_project\train.py","time_series_prediction_project\predict.py","time_series_prediction_project\requirements.txt","time_series_prediction_project\README.md"



# Key Notes:

**Modular design:**

- Switch between RNN, LSTM, GRU in rnn_model.py
- Train once, test/predict easily
- Sequence length: Adjust seq_length depending on the data

- Scalers: Always normalize for RNN training
- Visualization: Easy plotting using utils/visualization.py