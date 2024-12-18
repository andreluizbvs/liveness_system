# The reson for some commented code in this file is at the function "augment"

# import random

import albumentations as A
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    RandomCrop,
    RandomBrightness,
    RandomContrast,
    GaussianNoise,
)

# from src.models.adversarial_attack import AdversarialAttack

# Constants
SEED_VALUE = 42
BATCH_SIZE = 32
IMG_SIZE = 224
dummy_image = tf.zeros([1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)


def extract_face(image):
    try:
        image = image.numpy()
        image = image.astype(np.uint8)
        image = image[0]

        faces = DeepFace.extract_faces(
            image, detector_backend="yolov8", enforce_detection=False
        )
        if len(faces) > 0 and faces[0]["confidence"] > 0.5:
            bbox = faces[0]["facial_area"].astype(int)
            x1, y1, w, h = bbox
            face = image[y1 : y1 + h, x1 : x1 + w]
            return tf.expand_dims(
                tf.convert_to_tensor(face, dtype=tf.float32), 0
            )
        return dummy_image
    except Exception as e:
        print(f"Error in extract_face: {e}")
        return dummy_image


def preprocess_data(
    train_dataset,
    val_dataset,
    test_dataset,
    image_size,
    combine_frame_and_face=False,
):
    # Define traditional data augmentation pipeline (TF + Albumentations)
    data_augmentation = tf.keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.2),
            RandomZoom(0.2),
            RandomTranslation(0.1, 0.1),
            RandomCrop(height=image_size[0], width=image_size[1]),
            RandomBrightness(0.1),
            RandomContrast(0.1),
            GaussianNoise(0.1),
        ]
    )
    # alb_augs = A.Compose([A.MotionBlur(p=0.1), A.CoarseDropout(p=0.1)])

    # The explanation of why this is commented out is right in the comment below
    # (Actually, this is directly applied in the training script).
    # adv_att = AdversarialAttack()
    # manipulate_img = adv_att.manipulate_image
    # attack_prob = 0.2

    def augment(image, label):
        # Applying traditional data augmentation
        image = data_augmentation(image, training=True)
        # image = tf.numpy_function(
        #     lambda img: alb_augs(image=img)["image"], [image], tf.float32
        # )
        # ****** Applying anti-spoof-focused data augmentation ******
        # To test the model against adversarial attacks, we can't apply 
        # adversarial attack data augmentation to the training set, only to 
        # the testset. After obtaining the evaluation metrics, we can add the 
        # adv. attack data aug. to the training set and retrain the model to 
        # see if the it became robust to them.
        #
        # if random.random() < attack_prob:
        #     image = tf.numpy_function(
        #         lambda img: manipulate_img(img.numpy()), [image], tf.float32
        #     )
        # image.set_shape(image_size + (3,))
        return image, label

    train_dataset = train_dataset.map(augment)

    if combine_frame_and_face:
        print("Combining frame and face...")

        # Extract faces and create dual input pipeline
        def preprocess(image, label):
            face = tf.py_function(extract_face, [image], tf.float32)
            face = tf.ensure_shape(face, [1, image_size[0], image_size[1], 3])
            return (image, face), label
    else:
        print("Not combining frame and face...")

        def preprocess(image, label):
            image /= 127.5
            image -= 1.0
            if image.shape[0] != image_size[0]:
                image = image[0]
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
    data_dir,
    image_size,
    batch_size=1,
    test_size=0.3,
    combine_frame_and_face=False,
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

    return preprocess_data(
        train_dataset,
        val_dataset,
        test_dataset,
        image_size,
        combine_frame_and_face,
    )


def create_dataset_from_split(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
    image_size=IMG_SIZE,
    combine_frame_and_face=False,
):
    # Ensure data is tf tensor floating point type
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float16)
    X_valid = tf.convert_to_tensor(X_valid, dtype=tf.float16)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float16)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float16)
    y_valid = tf.convert_to_tensor(y_valid, dtype=tf.float16)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float16)

    # Print shapes and data types for debugging
    print("X_train shape:", X_train.shape, "dtype:", X_train.dtype)
    print("X_valid shape:", X_valid.shape, "dtype:", X_valid.dtype)
    print("X_test shape:", X_test.shape, "dtype:", X_test.dtype)
    print("y_train shape:", y_train.shape, "dtype:", y_train.dtype)
    print("y_valid shape:", y_valid.shape, "dtype:", y_valid.dtype)
    print("y_test shape:", y_test.shape, "dtype:", y_test.dtype)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    print("Preprocessing data...")
    return preprocess_data(
        train_dataset,
        val_dataset,
        test_dataset,
        image_size,
        combine_frame_and_face,
    )
