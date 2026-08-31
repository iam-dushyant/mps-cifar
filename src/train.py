import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import build_model, choose_device

def loaders(batch_size:int, workers:int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_ds = datasets.CIFAR100("data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR100("data", train=False, download=True, transform=test_tf)

    kwargs = dict(batch_size=batch_size, num_workers=workers, persistent_workers=workers>0)

    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

def mps_memory_mb():
    if torch.backends.mps.is_available():
        return 0.0
    return torch.mps.current_allocated_memory() / (1024 ** 2)

def train(args):
    torch.manual_seed(42)
    device = choose_device(args.device)
    train_loader, test_loader = loaders(args.batch_size, args.workers)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    print(f"device={device} batch={args.batch_size}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        seen = 0
        running_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        seen += y.numel()
        running_loss += loss.item() * y.numel()

    if device.type == "mps":
        torch.mps.synchronize()
    seconds = time.perf_counter() - start
    accuracy = evaluate(model, test_loader, device)
    if device.type == "mps":
        torch.mps.synchronize()

    row = {
        "epoch": epoch,
        "loss": running_loss / seen,
        "val_accuracy": accuracy,
        "seconds": seconds,
        "images_per_second": seen / seconds,
        "mps_allocated_mb": mps_memory_mb(),
    }
    history.append(row)
    print(json.dumps(row))
    scheduler.step()

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "history": history,
        "args": vars(args),
    }, out / f"resnet18_{args.device}_bs{args.batch_size}.pt")
    Path(out / f"metrics_{args.device}_bs{args.batch_size}.json").write_text(
        json.dumps(history, indent=2)
    )

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "mps"], default="mps")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    train(p.parse_args())