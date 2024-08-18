import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    RandomCrop,
)

seed_value = 42


def moire_pattern(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a repetitive pattern
    rows, cols = gray_image.shape
    pattern = np.zeros((rows, cols), dtype=np.uint8)

    # Define the frequency and amplitude of the pattern
    frequency = 10
    amplitude = 50

    for i in range(rows):
        for j in range(cols):
            pattern[i, j] = 128 + amplitude * np.sin(
                2 * np.pi * frequency * (i + j) / cols
            )

    # Blend the pattern with the original image
    moire_image = cv2.addWeighted(gray_image, 0.5, pattern, 0.5, 0)
    del gray_image, pattern

    # Convert back to BGR
    moire_image = cv2.cvtColor(moire_image, cv2.COLOR_GRAY2BGR)

    return moire_image


def color_distortion(image):
    # Apply color distortion effect to the image
    # Implementation code goes here
    return image


def facial_artifacts(image):
    # Apply facial artifacts effect to the image
    # Implementation code goes here
    return image


def gradient_noise(image):
    # Apply gradient noise effect to the image
    # Implementation code goes here
    return image


def apply_custom_augmentation(image):
    augmentations = [
        moire_pattern,
        color_distortion,
        facial_artifacts,
        gradient_noise,
    ]
    augmentation = random.choice(augmentations)
    return augmentation(image)


def create_dataset(data_dir, image_size, batch_size=32, test_size=0.3):
    # Load the dataset from the directory
    train_dataset = image_dataset_from_directory(
        data_dir,
        validation_split=test_size,
        subset="training",
        seed=seed_value,
        image_size=image_size,
        batch_size=batch_size,
    )

    test_dataset = image_dataset_from_directory(
        data_dir,
        validation_split=test_size,
        subset="validation",
        seed=seed_value,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Define data augmentation pipeline
    data_augmentation = tf.keras.Sequential(
        [
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.2),
            RandomZoom(0.2),
            RandomTranslation(0.1, 0.1),
            RandomCrop(height=image_size[0], width=image_size[1]),
        ]
    )

    # Print class names and their corresponding indices
    class_names = train_dataset.class_names
    print("Class names and their corresponding indices:", class_names)

    # Split test_dataset into validation and test datasets
    val_size = int(len(test_dataset) * (1 / 3))
    test_size = len(test_dataset) - val_size

    val_dataset = test_dataset.take(val_size)
    test_dataset = test_dataset.skip(val_size)

    # Apply data augmentation to the training dataset
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    )

    # Prefetch the datasets for performance optimization
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
