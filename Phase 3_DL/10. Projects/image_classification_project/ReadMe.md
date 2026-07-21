# folder structure with all directories and files:

image_classification_project/
│
├── data/                     # Your datasets
│   ├── cats_dogs/
│   │    ├── cat/
│   │    └── dog/
│   └── leaf_dataset/
│        ├── healthy/
│        ├── infected/
│        └── nutrient_deficient/
│
├── models/                   # Trained models will be saved here
│
├── utils/
│   └── data_utils.py         # Data generator functions
│
├── cnn_model.py              # Custom CNN model
├── mobilenet_model.py        # Transfer learning model
├── train.py                  # Training script
├── predict.py                # Prediction script for new images
└── requirements.txt          # Packages
└── README.md                 # Project documentation

# pip install -r requirements.txt


# After training
np.save('models/class_indices.npy', train_gen.class_indices)


# Powerahell
New-Item -ItemType Directory -Path "image_classification_project\data\cats_dogs\cat","image_classification_project\data\cats_dogs\dog","image_classification_project\data\leaf_dataset\healthy","image_classification_project\data\leaf_dataset\infected","image_classification_project\data\leaf_dataset\nutrient_deficient","image_classification_project\models","image_classification_project\utils"; New-Item -ItemType File -Path "image_classification_project\utils\data_utils.py","image_classification_project\cnn_model.py","image_classification_project\mobilenet_model.py","image_classification_project\train.py","image_classification_project\predict.py","image_classification_project\requirements.txt"




# Step 1: Run train.py
python train.py

**What it does:**

- Imports the data generators from utils/data_utils.py.
- Loads the dataset from your directory-style folders (data/cats_dogs or data/leaf_dataset).
- Trains the custom CNN and MobileNet models sequentially.
- Saves the trained models in the models/ folder:

1. custom_cnn.h5
2. mobilenet_model.h5

- Saves class indices (class_indices.npy) for mapping predictions.

# Step 2: Run predict.py
python predict.py


**What it does:**

- Loads a trained model from models/ (custom_cnn.h5 or mobilenet_model.h5).
- Loads the saved class mapping (class_indices.npy).
- Loads and preprocesses a single image.
- Predicts the class and prints it.

**Important notes**

- You don’t run cnn_model.py, mobilenet_model.py, or data_utils.py directly.
- They are modules that train.py imports.
- train.py must be run first, otherwise the models and class indices won’t exist for prediction.

**✅ Workflow summary:**

- Prepare your dataset in directory-style folders.
- Run train.py → trains and saves models.
- Run predict.py → predicts new images using saved models.