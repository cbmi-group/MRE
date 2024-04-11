# -*- coding: utf-8 -*-

import cv2
import numpy as np
from random import randint, random, randrange
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def random_flip(image, mask):
    flip_seed = randint(-1,2)
    print(flip_seed)
    if flip_seed != 2:
        image = cv2.flip(image, flip_seed)
        mask = cv2.flip(mask, flip_seed)
    return image, mask

def flip(image, mask):
    flip_seed = 1

    image = cv2.flip(image, flip_seed)
    mask = cv2.flip(mask, flip_seed)
    return image, mask


def rotation_image(image, mask, angle=0, scale=1):
    height, width = image.shape
    M = cv2.getRotationMatrix2D((height/2,width/2), angle, scale)
    out_image = cv2.warpAffine(image, M, (height,width))
    out_mask = cv2.warpAffine(mask, M, (height,width), flags=cv2.INTER_NEAREST, borderValue=(255,255,255))
    return out_image, out_mask


def random_rotation_scale(image, mask, angle_min=-45, angle_max=45, scale_var=True):
    if scale_var:
        scale = randint(80, 120) / 100. # scale from 0.8-1.2
    else:
        scale = 1
    if angle_min < angle_max:
        angle = randint(angle_min, angle_max) # rotation angle from [-45, 45]
    out_image, out_mask = rotation_image(image, mask, angle=angle, scale=scale)
    return out_image, out_mask


def random_shift(image, mask, shift_range=20):
    rand_seed_1 = -1 if random() < 0.5 else 1
    trans_x = rand_seed_1 * random() * shift_range # shift direction, shift percentage based on shift range
    rand_seed_2 = -1 if random() < 0.5 else 1
    trans_y = rand_seed_2 * random() * shift_range

    M = np.array([[1,0,trans_x], [0,1,trans_y]], dtype=np.float32)
    out_image = cv2.warpAffine(image, M, image.shape)
    out_mask = cv2.warpAffine(mask, M, mask.shape, flags=cv2.INTER_NEAREST, borderValue=(255,255,255))
    return out_image, out_mask


def random_shear(image, mask, shear_range=0.2):
    rand_seed_1 = -1 if random() < 0.5 else 1
    shear_factor = rand_seed_1 * random() * shear_range

    w, h = image.shape[1], image.shape[0]
    if shear_factor < 0:
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
    M = np.array([[1, abs(shear_factor), 0],
                  [0, 1,                 0]])
    nW = w + abs(shear_factor * h)
    image = cv2.warpAffine(image, M, (int(nW), h))
    mask = cv2.warpAffine(mask, M, (int(nW), h), flags=cv2.INTER_NEAREST, borderValue=(255,255,255))
    if shear_factor < 0:
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
    image_out, mask_out = cv2.resize(image, (w,h)), cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
    return image_out, mask_out


def random_contrast(image):
    factor = randint(7,10) / 10
    mean = np.uint16(np.mean(image) + 0.5)
    mean_img = (np.ones(image.shape) * mean).astype(np.uint16)
    out_image = image.astype(np.uint16) * factor + mean_img * (1.0 - factor) 
    if factor < 0 or factor > 1:
        out_image = clip_image(out_image.astype(np.float))
    return out_image.astype(np.uint16)


def random_brightness(image):
    # noise_direction = randint(-1,1)
    # noise_scale = 0.1 * (np.amax(image) - np.amin(image))
    # noise_seed = random()
    # noise_img = image + noise_direction * noise_scale * noise_seed
    noise_scale = randint(7,13) / 10.
    noise_img = image * noise_scale
    out_image = clip_image(noise_img)
    return out_image


def random_noise(image):
    noise_seed = randint(0,1)
    if noise_seed == 0:
        noise_img = cv2.GaussianBlur(image, (5,5), 0)
    else:
        noise_img = image
    return noise_img


def clip_image(image):
    image[image > 65535.] = 65535
    image[image < 0.] = 0
    image = image.astype(np.uint16)
    return image


def convert_mask(mask):
    mask[mask >= 127.5] = 255
    mask[mask < 127.5] = 0 
    mask = mask.astype(np.uint8)
    return mask

