import cv2
import numpy as np

from src.models.adversarial_attack import AdversarialAttack

# img = cv2.imread('/home/andreluizbvs/Documents/authorized_faces/Screenshot from 2024-08-10 04-18-43.png')
img = cv2.imread('../data/path/to/image/')

adv_att = AdversarialAttack()

img_aug = adv_att.manipulate_image(img)

mask = np.dstack([adv_att.mask] * 3)
img = cv2.resize(img, (640, 640))
# Display the original image and the mask
side_by_side = np.hstack((img, mask, img_aug))
cv2.imshow('Images Side by Side', side_by_side)
cv2.waitKey(0)
cv2.destroyAllWindows()