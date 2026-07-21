from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D


def build_mobilenet_model(
    input_shape=(224, 224, 3), num_classes=2, fine_tune_layers=20
):
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Fine-tune top layers
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
