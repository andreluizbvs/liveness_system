import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.preprocessing import image


class LivenessModel:
    def __init__(self, model_path=None):
        self.img_size = 224
        self.model = self.__build_model(model_path)

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

    def train(self, train_dataset, val_dataset, epochs=10, batch_size=32):
        checkpoint = ModelCheckpoint(
            "../ckpt/best_model.keras",
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

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def preprocess(self, img_path):
        img = image.load_img(
            img_path, target_size=(self.img_size, self.img_size)
        )
        img_array = image.img_to_array(img)
        return np.expand_dims(img_array, axis=0)
