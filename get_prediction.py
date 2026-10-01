import torch
from PIL import Image
from torchvision import transforms
from UNet import UNet
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


if __name__ == "__main__":
    model = load_model()
    #image = Image.open("1000006460.jpg").convert("RGB")
    image = Image.open("belle_lay.jpg").convert("RGB")
    transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(256),
        #transforms.RandomRotation((180,180)),
        ResizePad(320),
        transforms.Lambda(lambda img: TF.rotate(img, -90)),
        transforms.ToTensor()
        ])

    image = transform(image)
    get_prediction(model, image)

