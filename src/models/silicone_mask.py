
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Input,
    Concatenate,
)

from src.models.base_model import BaseModel


class SiliconeMaskModel(BaseModel):

    def __init__(
        self,
        model_path=None,
        img_size=BaseModel.img_size,
        best_weights_path="../ckpt/best_silicone_mask_model.keras",
        combine_frame_and_face=False,
    ):
        super().__init__(img_size, best_weights_path)
        self.model = (
            self._build_model_v2(model_path)
            if combine_frame_and_face
            else self._build_model(model_path)
        )


    def _build_model(self, model_path=None):
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
            print(f"Model loaded from {model_path}")

        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        return model

    def _build_model_v2(self, model_path=None):
        # Base model for the video frame
        base_model_frame = ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=Input(shape=(self.img_size, self.img_size, 3)),
            name="frame",
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
            name="face",
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
