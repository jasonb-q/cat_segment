from pycocotools.coco import COCO

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, InterpolationMode
from transforms import *

class CocoDataset(Dataset):
    def __init__(self, image_dir, annotation_file, image_size=256, augment=False):
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        self.coco = COCO(annotation_file)
        
        self.cat_ids = self.coco.getCatIds(catNms=["cat"])
        self.image_ids = self.coco.getImgIds(catIds=self.cat_ids)

        print(f"Found {len(self.image_ids)} COCO images containing cats")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info["file_name"])

        image = Image.open(image_path).convert("RGB")

        annotation_ids = self.coco.getAnnIds(
                imgIds=[image_id],
                catIds=self.cat_ids,
                iscrowd=None
                )

        annotations = self.coco.loadAnns(annotation_ids)

        mask = np.zeros(
                (
                    image_info["height"],
                    image_info["width"]
                ),
                dtype=np.uint8
        )

        for annotation in annotations:
            instance_mask = self.coco.annToMask(annotation)
            mask = np.logical_or(mask, instance_mask)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        

        image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR
        )

        mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST
        )

        # augmentations
        if self.augment:
            image, mask = r_affine(image, mask)
            image, mask = hflip(image, mask)
            image = r_gaussian(image)
            image = brightness(image)
            image = contrast(image)
            image = saturation(image)

        image = TF.to_tensor(image)
        mask = np.array(mask, dtype=np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask

