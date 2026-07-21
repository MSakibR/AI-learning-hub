import torch


def load_yolo_model(model_name="yolov5s", pretrained=True):
    """
    Load YOLO model from PyTorch Hub
    yolov5s, yolov5m, yolov5l, yolov5x
    """
    model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=pretrained)
    return model
