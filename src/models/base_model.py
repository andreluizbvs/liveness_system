from abc import ABC, abstractmethod
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf2onnx

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.preprocessing import image


IMG_SIZE = 224


class LossPlotterCallback(Callback):
    def __init__(self):
        super(LossPlotterCallback, self).__init__()
        self.train_losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        plt.figure()
        plt.plot(
            range(1, len(self.train_losses) + 1),
            self.train_losses,
            label="Training Loss",
        )
        plt.plot(
            range(1, len(self.val_losses) + 1),
            self.val_losses,
            label="Validation Loss",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Training and Validation Loss Epoch {epoch}")
        plt.savefig("loss_plot.png")

    def on_train_end(self):
        plt.figure()
        plt.plot(
            range(1, len(self.train_losses) + 1),
            self.train_losses,
            label="Training Loss",
        )
        plt.plot(
            range(1, len(self.val_losses) + 1),
            self.val_losses,
            label="Validation Loss",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Final Training and Validation Loss Plot")
        plt.savefig("final_loss_plot.png")
        plt.show()


class BaseModel(ABC):

    img_size = IMG_SIZE

    def __init__(self, img_size, best_weights_path):
        self.img_size = img_size
        self.best_weights_path = best_weights_path


    def __call__(self, X):
        return self.predict(X)


    @abstractmethod
    def _build_model(self):
        pass


    def load_weights(self, model_path):
        if model_path:
            self.model.load_weights(model_path)
            print(f"Model loaded from {model_path}")


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
        class_name = self.__class__.__name__.lower()
        export_path = os.path.join(export_path, f"{class_name}_model.onnx")
        model_proto, _ = tf2onnx.convert.from_keras(
            self.model, input_signature=spec, output_path=export_path
        )
        print("Model converted to ONNX format:", export_path)
        return model_proto