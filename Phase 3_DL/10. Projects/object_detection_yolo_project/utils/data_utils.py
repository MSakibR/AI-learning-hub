import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_image(image_path):
    """Load image in RGB format"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def draw_boxes(image, boxes, labels, scores):
    """Draw bounding boxes on image"""
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return image
