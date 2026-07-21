from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_image_generators(
    data_dir, target_size=(224, 224), batch_size=32, val_split=0.2
):
    """
    Returns train and validation generators from a directory-style dataset.
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=val_split,
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
    )

    return train_gen, val_gen
