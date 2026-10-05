import torch
from PIL import Image
from torchvision import transforms
from UNet import UNet
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
from CatSegDataset import CatSegDataset
from CocoDataset import CocoDataset
from torchvision.datasets import OxfordIIITPet
from torchvision.utils import save_image
from pathlib import Path
from torch.utils.data import Subset, random_split, DataLoader, ConcatDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAT_LABEL = 0

class ResizePad:
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        w, h = img.size

        scale = self.size / max(w, h)

        new_w = round(w * scale)
        new_h = round(h * scale)

        img = TF.resize(
            img,
            [new_h, new_w],
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        )

        pad_w = self.size - new_w
        pad_h = self.size - new_h

        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img = TF.pad(
            img,
            [left, top, right, bottom],
            fill=0
        )

        return img


def load_model():
    model = UNet().to(device)
    model.load_state_dict(torch.load("best_cat_unet.pt", map_location=device))
    return model 

def get_data_loaders(dataset, train_val=0.8, batch_size=16):
    train_size = int(train_val * len(dataset))
    val_size = len(dataset) - train_size

    train_d, val_d = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
            train_d,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
            )
    val_loader = DataLoader(
            val_d,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
            )
    return train_loader, val_loader

def get_prediction(model, image):
    print("checking best model")

    model.eval()
    
    x = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probability = torch.sigmoid(logits)
        prediction = (probability > 0.2).float()
        #prediction = (prediction.squeeze().cpu().numpy())

    prediction = prediction.squeeze(0).cpu()

    cutout = image * prediction
    cutout_pil = TF.to_pil_image(cutout)
    alpha = (prediction.squeeze(0) * 255).byte()
    alpha_pil = TF.to_pil_image(alpha)

    cutout_pil = cutout_pil.convert("RGBA")
    cutout_pil.putalpha(alpha_pil)

    cutout_pil.save("belle_lay_cut.png")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1,3,1)
    plt.imshow(cutout.permute(1,2,0))
    plt.axis("off")
    plt.show()
    #plt.subplot(1,3,1)
    #plt.imshow(image.permute(1,2,0))
    #plt.title("image")
    #plt.axis("off")
#
    #plt.subplot(1,3,2)
    #plt.imshow(prediction, cmap="gray")
    #plt.title("Prediction")
    #plt.axis("off")
    #plt.show()

def get_cat_indicies(dataset):
    #yep, I can't spell apparently. As a joke I am going to leave it.
    cat_indecies = []
    for i in range(len(dataset)):
        image, (species, mask) = dataset[i]
        if species == CAT_LABEL:
            cat_indecies.append(i)
    return cat_indecies

def export_dataset(loader, split):
    root = Path("/home/qwaddles/Documents/cat_image_harmonization/data") / split

    image_dir = root / "images"
    mask_dir = root / "masks"

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)


    index = 0
    for images, masks in loader:
        for image, mask in zip(images, masks):
            save_image(
                    image,
                    image_dir / f"{index:06d}.png"
            )
            save_image(
                    mask,
                    mask_dir / f"{index:06d}.png"
            )
            index += 1
    print(f"Saved {index} samples to {root}")

if __name__ == "__main__":
    model = load_model()
    #image = Image.open("1000006460.jpg").convert("RGB")

    co_dataset = CocoDataset(
            "/home/qwaddles/Documents/cat_segment/data/coco2017/train2017",
            "/home/qwaddles/Documents/cat_segment/data/coco2017/annotations/instances_train2017.json",
            image_size=320,
            augment=False
    )
    
    oxford = OxfordIIITPet(
            root="./data",
            split="trainval",
            target_types=["binary-category", "segmentation"],
            download=True,
            )

    cat_indecies = get_cat_indicies(oxford)
    cats_d = CatSegDataset(oxford, cat_indecies, False, 320)
    train_d = ConcatDataset([cats_d, co_dataset])

    train_loader, val_loader = get_data_loaders(train_d, batch_size=8)
    export_dataset(train_loader, "train")
    export_dataset(val_loader, "val")



