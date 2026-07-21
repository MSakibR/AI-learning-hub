from utils.data_utils import create_image_generators
from cnn_model import build_custom_cnn
from mobilenet_model import build_mobilenet_model

# Dataset path (change as needed)
data_dir = "data/cats_dogs"

# Create data generators
train_gen, val_gen = create_image_generators(
    data_dir, target_size=(224, 224), batch_size=32
)

# -------------------------
# Train Custom CNN
# -------------------------
cnn_model = build_custom_cnn(
    input_shape=(224, 224, 3), num_classes=train_gen.num_classes
)
cnn_model.fit(train_gen, validation_data=val_gen, epochs=10)
cnn_model.save("models/custom_cnn.h5")

# -------------------------
# Train MobileNet Transfer Learning
# -------------------------
mobilenet_model = build_mobilenet_model(
    input_shape=(224, 224, 3), num_classes=train_gen.num_classes
)
mobilenet_model.fit(train_gen, validation_data=val_gen, epochs=10)
mobilenet_model.save("models/mobilenet_model.h5")
