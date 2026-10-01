import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Subset, random_split, DataLoader, ConcatDataset
from CatSegDataset import CatSegDataset
from CocoDataset import CocoDataset
from UNet import UNet
from tqdm import tqdm

CAT_LABEL = 0
BACKGROUND = 2
BOUNDING = 3
PET = 1
bce = nn.BCEWithLogitsLoss()

def get_cat_indicies(dataset):
    #yep, I can't spell apparently. As a joke I am going to leave it.
    cat_indecies = []
    for i in range(len(dataset)):
        image, (species, mask) = dataset[i]
        if species == CAT_LABEL:
            cat_indecies.append(i)
    return cat_indecies

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

def dice_loss(logits, targets, smooth=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.flatten(1)
    targets = targets.flatten(1)

    intersection = (probs * targets).sum(dim=1)
    dice = ( 2 * intersection + smooth ) / (probs.sum(dim=1) + targets.sum(dim=1) + smooth)
    return 1-dice.mean()

def loss_fn(logits, targets):
    bce_loss = bce(logits, targets)
    d_loss = dice_loss(logits, targets)
    return bce_loss + d_loss


def train_one_epoch( model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def dice_score(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)

    preds = (probs > threshold).float()
    preds = preds.flatten(1)
    targets = targets.flatten(1)

    intersection = (preds * targets).sum(dim=1)

    dice = (2 * intersection + 1e-6) / (preds.sum(dim=1) + targets.sum(dim=1) + 1e-6)
    return dice.mean()

def validate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = loss_fn(logits, masks)
            dice = dice_score(logits, masks)
            total_loss += loss.item()
            total_dice += dice.item()
    return (total_loss / len(loader), total_dice / len(loader))

def check_model(model, image, gt_mask):
    print("checking best model")
    model.load_state_dict(torch.load("best_cat_unet.pt", map_location=device))
    model.eval()
    
    x = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probability = torch.sigmoid(logits)
        prediction = (probability > 0.5).float()
        prediction = (prediction.squeeze().cpu().numpy())

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1,3,1)
    plt.imshow(image.permute(1,2,0))
    plt.title("image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask.squeeze(), cmap="gray")
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(prediction, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")
    plt.show()

def train(model, train_loader, val_loader, epochs):
    #best dice using dice 0.8568
    best_dice = 0.0
    best_loss = float("inf")
    patience = 10
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    val_dices = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6
            )

    for epoch in range(epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_cat_unet.pt")
            print("yippi, new high score!")
        else:
            epochs_without_improvement += 1
        
        print(
                f"Epoch: {epoch + 1:02d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"LR: {current_lr:.2e}"
                )

        if epochs_without_improvement >= patience:
            print("Early Stopping")
            break

    
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    co_dataset = CocoDataset(
            "/home/qwaddles/Documents/cat_segment/data/coco2017/train2017",
            "/home/qwaddles/Documents/cat_segment/data/coco2017/annotations/instances_train2017.json",
            image_size=320,
            augment=True
    )
    
    oxford = OxfordIIITPet(
            root="./data",
            split="trainval",
            target_types=["binary-category", "segmentation"],
            download=True,
            )

    cat_indecies = get_cat_indicies(oxford)
    cats_d = CatSegDataset(oxford, cat_indecies, 320)
    train_d = ConcatDataset([cats_d, co_dataset])
    train_loader, val_loader = get_data_loaders(train_d, batch_size=8)
#
    model = UNet().to(device)
    epochs = 100
    train(model, train_loader, val_loader, epochs)
    images_v, masks_v = next(iter(val_loader))
    check_model(model, images_v[0], masks_v[0])
