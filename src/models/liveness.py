import numpy as np

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.preprocessing import image

class LivenessModel:
    def __init__(self):
        self.model = self.build_model()

    def __call__(self, X):
        return self.model.predict(X)

    def build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "precision", "recall"],
        )
        return model

    def train(self, train_dataset, test_dataset, epochs=10, batch_size=32):
        checkpoint = ModelCheckpoint(
            '../ckpt/best_model.keras', 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        )

        self.model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint]
        )

    @staticmethod
    def preprocess(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        return np.expand_dims(img_array, axis=0)
