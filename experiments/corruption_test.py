import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from models.model_loader import load_model
torch.backends.cudnn.benchmark = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_gaussian_noise(img, sigma):
    img = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma * 255, img.shape)
    img = img + noise
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return Image.fromarray(img)


def apply_motion_blur(img, radius=5):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness_shift(img, factor):
    return F.adjust_brightness(img, factor)


class CorruptedDataset(torch.utils.data.Dataset):

    def __init__(self, root, corruption=None, level=None):

        self.dataset = ImageFolder(root)

        self.corruption = corruption
        self.level = level

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img, label = self.dataset[idx]

        if self.corruption == "gaussian":
            img = add_gaussian_noise(img, self.level)

        elif self.corruption == "motion":
            img = apply_motion_blur(img, radius=self.level)

        elif self.corruption == "brightness":
            img = apply_brightness_shift(img, self.level)

        img = self.transform(img)

        return img, label


def evaluate_model(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def run_corruption_tests():

    dataset_path = "dataset"

    batch_size = 64

    results = []

    models = [
        ("resnet50", "resnet50_full_finetune.pth"),
        ("densenet121", "densenet121_full_finetune.pth"),
        ("efficientnet_b0", "efficientnet_b0_last_block.pth")
    ]

    corruptions = {

        "gaussian": [0.05, 0.1, 0.2],
        "motion": [3,5],
        "brightness": [0.5,1.5]

    }

    for model_name, weight in models:

        print("\nEvaluating:", model_name)

        model = load_model(model_name, num_classes=30)

        model.load_state_dict(
            torch.load(weight, map_location=device)
        )

        model.to(device)

        clean_dataset = CorruptedDataset(dataset_path)
        clean_loader = DataLoader(clean_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

        clean_acc = evaluate_model(model, clean_loader)

        print("Clean accuracy:", clean_acc)

        for corruption in corruptions:

            for level in corruptions[corruption]:

                dataset = CorruptedDataset(dataset_path, corruption, level)

                loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

                acc = evaluate_model(model, loader)

                corruption_error = 1 - acc

                relative_robustness = acc / clean_acc

                results.append({

                    "model": model_name,
                    "corruption": corruption,
                    "level": level,
                    "accuracy": acc,
                    "corruption_error": corruption_error,
                    "relative_robustness": relative_robustness
                })

                print(f"{model_name} | {corruption} | level {level} | accuracy {acc:.4f}")

    df = pd.DataFrame(results)

    df.to_csv("corruption_results.csv", index=False)

    print("\nResults saved locally")


if __name__ == "__main__":
    run_corruption_tests()