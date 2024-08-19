import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Input,
    Concatenate,
)
from tensorflow.keras.preprocessing import image


class LivenessModel:
    def __init__(
        self,
        model_path=None,
        img_size=224,
        best_weights_path="../ckpt/best_weights.keras",
        combine_frame_and_face=False,
    ):
        self.img_size = img_size
        self.best_weights_path = best_weights_path
        self.model = (
            self.__build_model_v2(model_path)
            if combine_frame_and_face
            else self.__build_model(model_path)
        )

    def __call__(self, X):
        return self.predict(X)

    def __build_model(self, model_path=None):
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=(self.img_size, self.img_size, 3)),
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        predictions = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        if model_path:
            model.load_weights(model_path)

        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )
        return model

    def __build_model_v2(self, model_path=None):
        # Base model for the video frame
        base_model_frame = ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=(self.img_size, self.img_size, 3)),
            name="frame"
        )
        for layer in base_model_frame.layers:
            layer.name = f"frame_{layer.name}"
        x_frame = base_model_frame.output
        x_frame = GlobalAveragePooling2D()(x_frame)

        # Base model for extracted face from video frame
        base_model_face = ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=(self.img_size, self.img_size, 3)),
            name="face"
        )
        for layer in base_model_face.layers:
            layer.name = f"face_{layer.name}"
        
        x_face = base_model_face.output
        x_face = GlobalAveragePooling2D()(x_face)

        # Combine features
        combined = Concatenate()([x_frame, x_face])
        x = Dense(256, activation="relu")(combined)
        predictions = Dense(1, activation="sigmoid")(x)
        model = Model(
            inputs=[base_model_frame.input, base_model_face.input],
            outputs=predictions,
        )

        if model_path:
            model.load_weights(model_path)

        # Freeze the base model layers
        for layer in base_model_frame.layers:
            layer.trainable = False
        for layer in base_model_face.layers:
            layer.trainable = False

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )
        return model

    def train(self, train_dataset, val_dataset, epochs=10, batch_size=32):
        checkpoint = ModelCheckpoint(
            self.best_weights_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )

        self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint],
        )

    def evaluate(self, test_dataset, weights_path=None):
        if not weights_path:
            weights_path = self.best_weights_path
        self.model.load_weights(weights_path)
        return self.model.evaluate(test_dataset, return_dict=True)

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def preprocess(self, img_path):
        img = image.load_img(
            img_path, target_size=(self.img_size, self.img_size)
        )
        img_array = image.img_to_array(img)
        return np.expand_dims(img_array, axis=0)
