import cv2
import matplotlib.pyplot as plt


def draw_boxes(image, boxes, labels, scores):
    """
    Draw bounding boxes, class labels, and confidence scores on an image.
    Args:
        image: numpy array (H, W, 3)
        boxes: list of [x1, y1, x2, y2]
        labels: list of class names
        scores: list of confidence scores
    Returns:
        image with drawn boxes
    """
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return image


def show_image(image, title="Image"):
    """Display image using matplotlib"""
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()
