import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models.model_loader import load_model
from utils.dataset import get_dataloaders
from training.train import train_one_epoch
from training.evaluate import evaluate

from experiments.fine_tune import (
    linear_probe,
    last_block_finetune,
    full_finetune,
    selective_20_percent
)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
train_loader, val_loader, num_classes = get_dataloaders("dataset")

models_to_run = [
    "resnet50",
    "densenet121",
    "efficientnet_b0"
]

strategies = {
    "linear_probe": linear_probe,
    "last_block": last_block_finetune,
    "full_finetune": full_finetune,
    "selective20": selective_20_percent
}

epochs = 15

for model_name in models_to_run:

    for strategy_name, strategy in strategies.items():

        print(f"\n=== {model_name} | {strategy_name} ===\n")

        model = load_model(model_name, num_classes)

        strategy(model)

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )

        train_accs = []
        val_accs = []
        train_losses = []
        grad_norms = []

        for epoch in range(epochs):

            train_loss, train_acc, grad_norm = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device
            )

            val_acc = evaluate(model, val_loader, device)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            grad_norms.append(grad_norm)

            print(
                f"Epoch {epoch+1} | "
                f"Loss {train_loss:.3f} | "
                f"Train {train_acc:.3f} | "
                f"Val {val_acc:.3f} | "
                f"GradNorm {grad_norm:.3f}"
            )

        torch.save(
            model.state_dict(),
            f"{model_name}_{strategy_name}.pth"
        )

        plt.figure()

        plt.plot(train_accs, label="train")
        plt.plot(val_accs, label="val")

        plt.legend()

        plt.title(f"{model_name} - {strategy_name} Accuracy")

        plt.savefig(
            f"{model_name}_{strategy_name}_accuracy.png"
        )

        plt.close()

        plt.figure()

        plt.plot(grad_norms)

        plt.title(f"{model_name} - {strategy_name} Gradient Norm")

        plt.savefig(
            f"{model_name}_{strategy_name}_gradnorm.png"
        )

        plt.close()