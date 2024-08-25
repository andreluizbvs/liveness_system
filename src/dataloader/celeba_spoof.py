import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

SEED_VALUE = 42
IMG_SIZE = 224


def get_ratio_bbox_and_image(full_img_path, bound_box_path):
    img = cv2.imread(full_img_path)
    real_h, real_w, _ = img.shape
    area_image = real_h * real_w
    _, _, w1, h1 = get_area_bbox_indices(bound_box_path, real_w, real_h)
    area_bbox = w1 * h1
    return area_bbox / area_image


def standard_width_height_scaling(real_w, real_h, bbox0, bbox1, bbox2, bbox3):
    x1 = int(int(bbox0) * (float(real_w) / 224))  # bbox[0]
    y1 = int(int(bbox1) * (float(real_h) / 224))  # bbox[1]
    w1 = int(int(bbox2) * (float(real_w) / 224))  # bbox[2]
    h1 = int(int(bbox3) * (float(real_h) / 224))  # bbox[3]
    return x1, y1, w1, h1


def get_area_bbox_indices(bound_box_path, real_w, real_h):
    bound_box_read = open(bound_box_path, "r")
    bound_box_indices = list()
    for i in bound_box_read:
        bound_box_indices.append(i)
    bbox = bound_box_indices[0].split()
    x1, y1, w1, h1 = standard_width_height_scaling(
        real_w, real_h, bbox[0], bbox[1], bbox[2], bbox[3]
    )
    return x1, y1, w1, h1


def get_padding_bbox_indices(
    x1, y1, w1, h1, real_w, real_h, ratio_bbox_and_image
):
    x1_padding = x1 - int((w1) * (1 + ratio_bbox_and_image))
    y1_padding = y1 - int((h1) * (1 + ratio_bbox_and_image))
    w1_padding = w1 + int((w1) * (1 + ratio_bbox_and_image))
    h1_padding = h1 + int((h1) * (1 + ratio_bbox_and_image))
    if x1_padding < 0:
        x1_padding = 0
    if y1_padding < 0:
        y1_padding = 0
    if w1_padding > real_w:
        w1_padding = real_w
    if h1_padding > real_h:
        h1_padding = real_h
    return x1_padding, y1_padding, w1_padding, h1_padding


def read_crop_img_with_bbox(full_img_path, bound_box_path):
    img = cv2.imread(full_img_path)
    real_w = img.shape[1]
    real_h = img.shape[0]
    x1, y1, w1, h1 = get_area_bbox_indices(bound_box_path, real_w, real_h)
    return x1, y1, w1, h1, img, real_w, real_h


def get_padding_cropped_img(rootdir, dim, count_limit_live, count_limit_spoof):
    count_live = 0
    count_spoof = 0
    padding_cropped_storage = []
    padding_cropped_labels = []
    img_names = []

    for file in os.listdir(rootdir):
        # file is 1, 1000, ..... 10029,...... => Name of folder
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            for e in os.listdir(d):
                # e is "live" of "spoof"
                imgs_path = d + "/" + e + "/"
                for img_path in os.listdir(imgs_path):
                    if img_path.endswith(".jpg") or img_path.endswith(".png"):
                        full_img_path = imgs_path + img_path
                        bound_box_path = full_img_path[0:-4] + "_BB.txt"
                        x1, y1, w1, h1, img, real_w, real_h = (
                            read_crop_img_with_bbox(
                                full_img_path, bound_box_path
                            )
                        )
                        ratio_bbox_and_image = get_ratio_bbox_and_image(
                            full_img_path, bound_box_path
                        )
                        x1_padding, y1_padding, w1_padding, h1_padding = (
                            get_padding_bbox_indices(
                                x1,
                                y1,
                                w1,
                                h1,
                                real_w,
                                real_h,
                                ratio_bbox_and_image,
                            )
                        )
                        padding_img = img[
                            y1_padding : y1 + h1_padding,
                            x1_padding : x1 + w1_padding,
                        ]
                        try:
                            if (
                                e == "live" and count_live >= count_limit_live
                            ) or (
                                e == "spoof"
                                and count_spoof >= count_limit_spoof
                            ):
                                continue
                            resized_padding_img = cv2.resize(
                                padding_img, dim, interpolation=cv2.INTER_AREA
                            )
                            padding_cropped_storage.append(resized_padding_img)
                            if e == "live":
                                count_live = count_live + 1
                                padding_cropped_labels.append(1)
                            elif e == "spoof":
                                count_spoof = count_spoof + 1
                                padding_cropped_labels.append(0)
                        except Exception as e:
                            continue

                        img_names.append(img_path)

                        if (count_live == count_limit_live and e == "live") or (
                            count_spoof == count_limit_spoof and e == "spoof"
                        ):
                            break
                if (
                    count_live >= count_limit_live
                    and count_spoof >= count_limit_spoof
                ):
                    break
        if count_live >= count_limit_live and count_spoof >= count_limit_spoof:
            print("DONE Extracting ")
            break

    print(f"Size of the dataset: {len(padding_cropped_storage)}")
    print(f"Size of the labels: {len(padding_cropped_labels)}")
    print(f"Shape of the image: {padding_cropped_storage[0].shape}")
    return padding_cropped_storage, padding_cropped_labels


def get_images_and_labels(
    padding_cropped_storage, padding_cropped_labels, mode="train"
):
    # Save the numpy to NUMPYZ
    X = np.asarray(padding_cropped_storage)
    y = np.asarray(padding_cropped_labels)
    np.savez(f"anti_spoofing_data_{mode}.npz", X, y)
    print("Data saved in npz file.")

    anti_spoofing_data = np.load(f"anti_spoofing_data_{mode}.npz")
    X, y = anti_spoofing_data["arr_0"], anti_spoofing_data["arr_1"]
    check_live_label = 0
    check_spoof_label = 0
    for i in y:
        if i == 1:
            check_live_label += 1
        elif i == 0:
            check_spoof_label += 1
    print(
        f"There are 2 classes. Number of lives is {check_live_label} "
        f"and number of spoofs is {check_spoof_label}"
    )
    return X, y


def get_data(lives=5000, spoofs=5000):
    # Live Storage

    dim = (IMG_SIZE, IMG_SIZE)
    rootdir_train = "../data/celebA-spoof/CelebA_Spoof_/CelebA_Spoof/Data/train"
    rootdir_test = "../data/celebA-spoof/CelebA_Spoof_/CelebA_Spoof/Data/test"

    test_proportion = 0.3
    train_lives = int(lives)
    train_spoofs = int(spoofs)
    test_lives = int(lives * test_proportion)
    test_spoofs = int(spoofs * test_proportion)

    padding_cropped_storage, padding_cropped_labels = get_padding_cropped_img(
        rootdir_train, dim, train_lives, train_spoofs
    )
    X_train, y_train = get_images_and_labels(
        padding_cropped_storage, padding_cropped_labels, mode="train"
    )

    test_padding_cropped_storage, test_padding_cropped_labels = (
        get_padding_cropped_img(rootdir_test, dim, test_lives, test_spoofs)
    )
    X_valid, y_valid = get_images_and_labels(
        test_padding_cropped_storage, test_padding_cropped_labels, mode="valid"
    )

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if i < 18:
            plt.imshow(X_train[i][:, :, ::-1])
        else:
            plt.imshow(X_valid[i - 18][:, :, ::-1])
    plt.show()

    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, test_size=test_proportion, random_state=SEED_VALUE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid, y_valid, test_size=0.5, random_state=SEED_VALUE
    )
    print(f"Training dataset size of X_train: {len(X_train)}")
    print(f"Testing dataset size of X_test: {len(X_test)}")
    print(f"Validation dataset size of X_valid: {len(X_valid)}")
    print(f"Testing dataset size of y_train: {len(y_train)}")
    print(f"Testing dataset size of y_test: {len(y_test)}")
    print(f"Testing dataset size of y_valid: {len(y_valid)}")

    return X_train, X_valid, X_test, y_train, y_valid, y_test
