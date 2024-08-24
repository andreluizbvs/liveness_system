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
from PIL import Image
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))


SEED_VALUE = 42
BATCH_SIZE = 32
dummy_image = tf.zeros([1, 224, 224, 3], dtype=tf.float32)

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

def extract_face(image):
    try:
        image = image.numpy()
        image = image.astype(np.uint8)
        image = image[0]
        
        faces = app.get(image)
        if len(faces) > 0 and faces[0].det_score > 0.5:
            bbox = faces[0].bbox.astype(int)
            x1, y1, x2, y2 = bbox
            face = image[int(y1):int(y2), int(x1):int(x2)]
            face = np.array(Image.fromarray(face, mode="RGB").resize((224, 224)))
            return tf.expand_dims(tf.convert_to_tensor(face, dtype=tf.float32), 0)
        
        # print("No face detected.")
        return dummy_image
    except Exception as e:
        print(f"Error in extract_face: {e}")
        return dummy_image


def preprocess_data(train_dataset, val_dataset, test_dataset, image_size, combine_frame_and_face=False):

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

    # Apply data augmentation to the training dataset
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    )

    if combine_frame_and_face:
        # Extract faces and create dual input pipeline
        def preprocess(image, label):
            face = tf.py_function(extract_face, [image], tf.float32)
            face = tf.ensure_shape(face, [1, image_size[0], image_size[1], 3])
            return (image, face), label
    else:
        def preprocess(image, label):
            image = image / 255.0
            image = image * 2.0 - 1.0
            return image, label

    train_dataset = train_dataset.map(preprocess)
    val_dataset = val_dataset.map(preprocess)
    test_dataset = test_dataset.map(preprocess)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Prefetch the datasets for performance optimization
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


def create_dataset(
    data_dir, image_size, batch_size=1, test_size=0.3, combine_frame_and_face=False
):
    # Load the dataset from the directory
    train_dataset = image_dataset_from_directory(
        data_dir,
        validation_split=test_size,
        subset="training",
        seed=SEED_VALUE,
        image_size=image_size,
        batch_size=batch_size,
    )

    test_dataset = image_dataset_from_directory(
        data_dir,
        validation_split=test_size,
        subset="validation",
        seed=SEED_VALUE,
        image_size=image_size,
        batch_size=batch_size,
    )

    # Print class names and their corresponding indices
    class_names = train_dataset.class_names
    print("Class names and their corresponding indices:", class_names)

    # Split test_dataset into validation and test datasets
    val_size = int(len(test_dataset) * (1.0 / 3.0))
    test_size = len(test_dataset) - val_size

    val_dataset = test_dataset.take(val_size)
    test_dataset = test_dataset.skip(val_size)

    return preprocess_data(train_dataset, val_dataset, test_dataset, image_size, combine_frame_and_face)


def create_dataset_from_split(
    X_train, X_valid, X_test, y_train, y_valid, y_test, image_size, combine_frame_and_face=False
):
    
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float16)
    X_valid = tf.convert_to_tensor(X_valid, dtype=tf.float16)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float16)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float16)
    y_valid = tf.convert_to_tensor(y_valid, dtype=tf.float16)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float16)

    print(X_train.shape, X_valid.shape, X_test.shape)
    print(y_train.shape, y_valid.shape, y_test.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return preprocess_data(train_dataset, val_dataset, test_dataset, image_size, combine_frame_and_face)
