import numpy as np
import torch
import random
from torchvision.transforms import functional as TF, InterpolationMode

def r_affine(image, mask):
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        translate = [
                random.randint(-15, -15),
                random.randint(-15, -15)
                ]
        scale = random.uniform(0.99, 1.1)
        image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0,0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0
                )
        mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0,0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0
                )
    return image, mask

def hflip(image, mask):
    if random.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    return image, mask

def r_gaussian(image):
    if random.random() < 0.2:
        image = TF.gaussian_blur(
                image,
                kernel_size=3,
                sigma=random.uniform(0.1, 1.0)
                )
    return image

def brightness(image):
    if random.random() < 0.5:
        image = TF.adjust_brightness(
                image,
                random.uniform(0.75, 1.25)
                )
    return image

def contrast(image):
    if random.random() < 0.5:
        image = TF.adjust_contrast(
                image,
                random.uniform(0.75, 1.25)
                )
    return image

def saturation(image):
    if random.random() < 0.3:
        image = TF.adjust_saturation(
                image,
                random.uniform(0.8, 1.2)
                )
    return image

    
    
    

