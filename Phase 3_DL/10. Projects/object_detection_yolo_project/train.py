from pathlib import Path
import torch

# YOLO requires labels in YOLO format: class x_center y_center width height (normalized)
# Suppose you have train/images and train/labels

data_yaml = {
    "train": "data/images/train",
    "val": "data/images/val",
    "nc": 3,  # number of classes
    "names": ["cat", "dog", "person"],
}

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
model.train(data="data.yaml", epochs=20, imgsz=640, batch_size=16)

# Save trained weights
model.save("models/yolov5_cats_dogs.pt")
