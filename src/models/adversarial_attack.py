import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src.dataloader.adversarial_attack_data_aug import add_moire_noise, digital_augment

IMG_SIZE =  224

app = FaceAnalysis()

app.prepare(ctx_id=0, det_size=(IMG_SIZE, IMG_SIZE))


class AdversarialAttack:
    def __init__(self, liveness_model = None):
        self.liveness_model = liveness_model
        self.adversarial_examples = []
        self.img_digital_aug = None
        self.img_moire = None
        self.mask = None

    def gen_binary_mask(self, image, landmarks):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        hull_points = cv2.convexHull(np.array(landmarks), returnPoints=True)
        cv2.fillPoly(mask, [hull_points], 255)
        self.mask = mask

    def manipulate_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = (image * 255).astype(np.uint8)
        faces = app.get(image)
        if len(faces) == 0:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3))

        # Create a binary face mask from the landmarks
        landmarks = faces[0]["landmark_2d_106"].astype(np.int32)
        self.gen_binary_mask(image, landmarks)

        # Apply the digital augmentations
        self.img_digital_aug = digital_augment(image, self.mask)

        # Add moire noise to the augmented image
        self.img_moire = add_moire_noise(image)

        manipulated_img = add_moire_noise(self.img_digital_aug)

        return cv2.resize(manipulated_img, (IMG_SIZE, IMG_SIZE))

    def generate_adversarial_examples(self, regular_images):
        self.adversarial_examples = []
        for regular_image in regular_images:
            image_aug = self.manipulate_image(regular_image)
            if image_aug is None:
                continue
            self.adversarial_examples.append(image_aug)

        return np.stack(self.adversarial_examples, axis=0)

    def test_adversarial_examples(self, images_to_manipulate: list = []):
        if self.liveness_model is None:
            print("Liveness model not found.")
            return None
        if len(images_to_manipulate) == 0:
            print("No images to augment.")
            return None
        if len(self.adversarial_examples) == 0:
            self.generate_adversarial_examples(images_to_manipulate)
        return self.liveness_model.predict(self.adversarial_examples)
