import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load model (custom CNN or MobileNet)
model_path = "models/mobilenet_model.h5"  # or 'custom_cnn.h5'
model = load_model(model_path)

# Load dataset info for class mapping
import pickle

class_indices = np.load("models/class_indices.npy", allow_pickle=True).item()
inv_class_map = {v: k for k, v in class_indices.items()}


# Predict function
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    pred = model.predict(x)
    class_index = np.argmax(pred)
    class_label = inv_class_map[class_index]
    print(f"Predicted Class: {class_label} ({pred[0][class_index]*100:.2f}%)")


# Example usage
predict_image("data/cats_dogs/cat/cat1.jpg")
