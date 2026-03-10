import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    grad_norm_epoch = 0

    scaler = torch.cuda.amp.GradScaler()

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        grad_norm = 0

        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().norm(2).item()

        grad_norm_epoch += grad_norm

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        _, preds = outputs.max(1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total

    avg_grad_norm = grad_norm_epoch / len(loader)

    return total_loss / len(loader), accuracy, avg_grad_norm