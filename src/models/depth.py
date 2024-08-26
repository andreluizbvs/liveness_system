import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Lambda,
    Layer
)

from src.models.base_model import BaseModel


class BackboneWrapper(Layer):
    def __init__(self, backbone, **kwargs):
        super(BackboneWrapper, self).__init__(**kwargs)
        self.backbone = backbone

    def call(self, inputs):
        return self.backbone(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": self.backbone,
            }
        )
        return config


class FaceDepthModel(BaseModel):
    def __init__(
        self,
        model_path=None,
        img_size=BaseModel.img_size,
        best_weights_path="../ckpt/best_face_depth_model.keras",
    ):
        super().__init__(img_size, best_weights_path)
        self.model = self._build_model(model_path)

    def _build_model(self, model_path=None):
        # Backbone to generate depth map
        backbone = from_pretrained_keras(
            "keras-io/monocular-depth-estimation",
            input_tensor=Input(shape=(self.img_size, self.img_size, 3)),
        )

        inputs = Input(shape=(self.img_size, self.img_size, 3))
        x = BackboneWrapper(backbone)(inputs)
        
        x = Lambda(
            lambda t:
            ((t - tf.reduce_min(t)) / (tf.reduce_max(t) - tf.reduce_min(t)))
        )(x)

        # Add the classification head
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        outputs = Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

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
