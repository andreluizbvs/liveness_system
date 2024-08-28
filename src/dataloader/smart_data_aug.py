import random

import albumentations as A
import cv2
import elasticdeform
import numpy as np
from albumentations import (
    ChannelShuffle,
    HueSaturationValue,
    RandomBrightnessContrast,
    Downscale,
    Sharpen,
    Resize,
    PadIfNeeded,
    CenterCrop,
    Affine,
)
from skimage.transform import PiecewiseAffineTransform, warp


# Moire effect simulation


def add_moire_noise(src):
    height, width = src.shape[:2]
    center = (height // 2, width // 2)
    degree = random.uniform(0.0005, 0.01)

    # Create grid coordinates
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Calculate offset from the center
    offset_x = x_grid - center[0]
    offset_y = y_grid - center[1]

    # Calculate angle and radius in polar coordinates
    theta = np.arctan2(offset_y, offset_x)
    radius = np.sqrt(offset_x**2 + offset_y**2)

    # Calculate new coordinates
    new_x = center[0] + radius * np.cos(theta + degree * radius)
    new_y = center[1] + radius * np.sin(theta + degree * radius)

    # Limit coordinate range
    new_x = np.clip(new_x, 0, width - 1).astype(np.int32)
    new_y = np.clip(new_y, 0, height - 1).astype(np.int32)

    # Apply moire effect
    dst = 0.8 * src + 0.2 * src[new_y, new_x]

    return dst.astype(np.uint8)



# =============================================================================


# Digital augmentation


def digital_augment(srcFace, mask_path, loc='all', mode='minor'):

    src_img, tar_img, spatial_transform = st_generator(srcFace, mode)

    # mask deformation
    hullMask = spatial_transform(image=mask_path)["image"]

    elt_mask = elasticdeform.deform_random_grid(hullMask, sigma=4, points=6)

    kernal_id = random.sample([0,1],2)
    kernal_sets = [(3,3),(5,5)]
    blur1 = cv2.GaussianBlur(elt_mask, kernal_sets[kernal_id[0]], 3)
    blured = cv2.GaussianBlur(blur1, kernal_sets[kernal_id[1]], 3)

    r = random.sample([0.25, 0.5, 0.75, 1, 1, 1], 1) if mode == 'minor' else random.sample([0.75, 1, 1, 1], 1)
    forge_mask = r*blured
    forge_mask = np.expand_dims(forge_mask, -1).repeat(3, axis=-1)

    bld_img = forge(src_img, tar_img, forge_mask)
    return bld_img
    

def transform_params(img, mode='minor'):

    if(mode=='minor'):
        color_transform=A.Compose([
            HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=10, always_apply=False, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, always_apply=False, p=0.5),
            Downscale(scale_min=0.35, scale_max=0.65, interpolation=0, always_apply=False, p=0.4),
            Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), always_apply=False, p=0.5),
            ])
    else:
        color_transform=A.Compose([
            ChannelShuffle(p=0.03),
            HueSaturationValue(hue_shift_limit=9, sat_shift_limit=9, val_shift_limit=20, always_apply=False, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
            Downscale(scale_min=0.25, scale_max=0.65, interpolation=0, always_apply=False, p=0.4),
            Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
            ])

    h, w = img.shape[:2]

    if (mode=='minor'):
        ampr, ampt = 0.02,0.02
    else:
        ampr, ampt = 0.04,0.04

    umin, umax = 1-ampr, 1+ampr
    uh,uw = random.uniform(umin, umax), random.uniform(umin, umax)
    hr, wr = int(uh*h), int(uw*w)   # resize

    vmin, vmax =  -ampt, ampt
    vh, vw = random.uniform(vmin, vmax), random.uniform(vmin, vmax)
    th, tw = int(vh*h), int(vw*w)   # translate

    spatial_transform=A.Compose([
        Resize(hr,wr),
        # LongestMaxSize(max_size=512, interpolation=1, always_apply=False, p=1),
        PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT,value=0),
        CenterCrop(height=h, width=w, always_apply=False, p=1.0),
        Affine(translate_px={'x':tw, 'y':th},  cval=0, mode=0, p=0.5)
    ])

    change_params = [hr, wr, th, tw]
    return color_transform, spatial_transform, change_params


def st_generator(src_img, mode):
    tar_img = src_img.copy()
    color_transform, spatial_transform, change_params = transform_params(src_img, mode)

    select_prob = random.random()
    flag = 0 if (select_prob < 0.5) else 1   # color change selection : "0-sr 1-tar"
    if(flag == 0):
        src_img = color_transform(image=src_img)["image"]
    else:
        tar_img = color_transform(image=tar_img)["image"]
    
    src_img = spatial_transform(image=src_img)["image"]

    return src_img, tar_img, spatial_transform


def forge(src_img, tar_img, mask):
    return (mask * src_img + (1 - mask) * tar_img).astype(np.uint8)


def random_deform(imageSize, nrows, ncols, mean=0, std=5):
    '''
    e.g. where nrows = 6, ncols = 7
    *_______*______*_____*______*______*_________*
    |                                            |
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *_______*______*_____*______*______*_________*
    '''
    h, w = imageSize
    rows = np.linspace(0, h-1, nrows).astype(np.int32)
    cols = np.linspace(0, w-1, ncols).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])
    return anchors, deformed.astype(np.int32)


def piecewise_affine_transform(image, srcAnchor, tgtAnchor):
    trans = PiecewiseAffineTransform()
    trans.estimate(srcAnchor, tgtAnchor)
    warped = warp(image, trans)
    return warped