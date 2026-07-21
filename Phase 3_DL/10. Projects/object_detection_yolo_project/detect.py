import torch
from utils.data_utils import draw_boxes, load_image
import cv2

# Load model
model = torch.hub.load(
    "ultralytics/yolov5", "custom", path="models/yolov5_cats_dogs.pt"
)

# Load test image
image_path = "data/test_images/test1.jpg"
img = load_image(image_path)

# Detect
results = model(img)
boxes = results.xyxy[0][:, :4].cpu().numpy()  # x1, y1, x2, y2
scores = results.xyxy[0][:, 4].cpu().numpy()  # confidence
labels = results.xyxy[0][:, 5].cpu().numpy()  # class indices

# Map indices to class names
class_names = ["cat", "dog", "person"]
labels = [class_names[int(i)] for i in labels]

# Draw boxes
output_img = draw_boxes(img, boxes, labels, scores)
cv2.imwrite("output_detected.jpg", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
