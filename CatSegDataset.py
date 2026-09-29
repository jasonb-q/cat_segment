import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, InterpolationMode

class CatSegDataset(Dataset):

    def __init__(self, dataset, indicies, image_size=256):
        self.dataset = dataset
        self.indicies = indicies
        self.image_size = image_size

    def __len__(self):
        return len(self.indicies)

    def __getitem__(self, idx):
        real_idx = self.indicies[idx]
        image, (_, mask) = self.dataset[real_idx]

        image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation = InterpolationMode.BILINEAR,
                )
        mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
                )
        image = TF.to_tensor(image)
        mask = np.array(mask)
        mask = (mask == 1).astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)
        return image, mask
