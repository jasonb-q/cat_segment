import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, InterpolationMode

class CocoNegativeDataset(Dataset):
    def __init__(self, coco, image_ids, image_dir, image_size=320, transform=None):
        self.coco = coco
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        info = self.coco.loadImgs(image_id)[0]

        path = os.path.join(self.image_dir, info["file_name"])
        image = Image.open(path).convert("RGB")
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)

        if self.transform is not None:
            image = self.transform(image)

        image = TF.to_tensor(image)

        mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        return image, mask
