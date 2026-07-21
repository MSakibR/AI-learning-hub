# folder structure with all directories and files:

object_detection_yolo_project/
│
├── data/                       
│   ├── images/                 # Images for training & testing
│   ├── labels/                 # YOLO format annotations (.txt)
│   └── test_images/            # Images for testing predictions
│
├── models/                     # Trained YOLO models
│
├── utils/
│   ├── data_utils.py           # Data loading & augmentation
│   └── visualization.py        # Functions to draw bounding boxes
│
├── yolo_model.py               # YOLOv5/YOLOv8 model setup
├── train.py                    # Training script
├── detect.py                   # Detection script for new images
├── requirements.txt            # Packages
└── README.md                   # Project documentation



# pip install -r requirements.txt



# 5️⃣ Key Notes for Object Detection Project

**YOLO format labels:**
Each image needs a .txt file with lines:

- class x_center y_center width height
- Values normalized [0,1].

**Pre-trained models:**
Use yolov5s for fast training, yolov5x for higher accuracy.

**Training Tips:**

- Use imgsz=640 or 416
- Batch size ~16–32
- Use data augmentation for better generalization

**Output:**

- Bounding boxes with labels and confidence scores
- Can be saved as images or video frames


# Powershell
New-Item -ItemType Directory -Path "object_detection_yolo_project\data\images","object_detection_yolo_project\data\labels","object_detection_yolo_project\data\test_images","object_detection_yolo_project\models","object_detection_yolo_project\utils"; New-Item -ItemType File -Path "object_detection_yolo_project\utils\data_utils.py","object_detection_yolo_project\utils\visualization.py","object_detection_yolo_project\yolo_model.py","object_detection_yolo_project\train.py","object_detection_yolo_project\detect.py","object_detection_yolo_project\requirements.txt","object_detection_yolo_project\README.md"


