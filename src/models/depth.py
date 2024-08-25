import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tf2onnx
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Input,
    Concatenate,
)
from silicone_mask import LossPlotterCallback


IMG_SIZE = 224


class FaceDepthModel:
    img_size = IMG_SIZE

    def __init__(
        self,
        model_path=None,
        img_size=IMG_SIZE,
        best_weights_path="../ckpt/best_model.keras",
    ):
        self.img_size = img_size
        self.best_weights_path = best_weights_path
        self.model = self.__build_model(model_path)

    def __call__(self, X):
        return self.predict(X)

    def __build_model(self, model_path=None):
        # TODO: Correct the backbone model
        backbone = hub.KerasLayer("https://tfhub.dev/google/monodepth2/1", trainable=False)

        # Add a classification head
        model = tf.keras.Sequential([
            backbone,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(1, activation='signoid')
        ])

        if model_path:
            model.load_weights(model_path)
            print(f"Model loaded from {model_path}")

        # Freeze the base model layers
        for layer in backbone.layers:
            layer.trainable = False

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        return model


    def train(self, train_dataset, val_dataset, epochs=10, patience=5):
        checkpoint = ModelCheckpoint(
            self.best_weights_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            mode="max",
            restore_best_weights=True,
        )
        loss_plotter = LossPlotterCallback()

        self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[checkpoint, early_stopping, loss_plotter],
        )

    def evaluate(self, test_dataset, weights_path=None):
        if not weights_path:
            weights_path = self.best_weights_path
        self.model.load_weights(weights_path)
        return self.model.evaluate(test_dataset, return_dict=True)

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    @classmethod
    def preprocess(cls, img_ref):
        if isinstance(img_ref, str):
            img_ref = image.load_img(
                img_ref, target_size=(cls.img_size, cls.img_size)
            )
            img_ref = image.img_to_array(img_ref)
        img_ref = cv2.resize(img_ref, (cls.img_size, cls.img_size))
        img_ref = (img_ref / 255.0).astype(np.float32)
        img_ref = img_ref * 2.0 - 1.0
        return np.expand_dims(img_ref, axis=0)

    def export_pb(self, export_path):
        self.model.export(export_path)
        print(f"Full TF Model exported to {export_path}")

    def export_onnx(self, export_path):
        spec = (
            tf.TensorSpec(
                (None, self.img_size, self.img_size, 3),
                tf.float32,
                name="input",
            ),
        )
        export_path = os.path.join(export_path, "face_depth_model.onnx")
        model_proto, _ = tf2onnx.convert.from_keras(
            self.model, input_signature=spec, output_path=export_path
        )
        print("Model converted to ONNX format:", export_path)
        return model_proto
