import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import build_model, choose_device


def loaders(batch_size: int, workers: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])

    train_ds = datasets.CIFAR100(
        "data",
        train=True,
        download=True,
        transform=train_tf,
    )

    test_ds = datasets.CIFAR100(
        "data",
        train=False,
        download=True,
        transform=test_tf,
    )

    kwargs = dict(
        batch_size=batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
    )

    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / total


def sync_device(device):
    if device.type == "mps":
        torch.mps.synchronize()

    elif device.type == "cuda":
        torch.cuda.synchronize()


def accelerator_memory_mb(device):
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2)

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    return 0.0


def train(args):
    torch.manual_seed(42)

    device = choose_device(args.device)

    train_loader, test_loader = loaders(
        args.batch_size,
        args.workers,
    )

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    print(f"device={device} batch={args.batch_size}")
    print(
        f"parameters="
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(f"AMP requested={args.amp}")

    if args.amp:
        print(
            f"AMP enabled for device type: {device.type}"
        )

    history = []

    for epoch in range(1, args.epochs + 1):

        model.train()

        start = time.perf_counter()

        seen = 0
        running_loss = 0.0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            amp_enabled = (
                args.amp
                and device.type in {"mps", "cuda"}
            )

            try:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(x)
                    loss = criterion(logits, y)

            except RuntimeError as exc:

                if device.type == "mps" and args.amp:
                    raise RuntimeError(
                        "\n"
                        "MPS AMP failed for this workload.\n"
                        "\n"
                        "Your current PyTorch/MPS combination "
                        "could not execute this model using "
                        "float16 autocast.\n"
                        "\n"
                        "Retry without --amp, for example:\n"
                        "\n"
                        f"python -m src.train "
                        f"--device mps "
                        f"--batch-size {args.batch_size} "
                        f"--epochs {args.epochs} "
                        f"--workers {args.workers}\n"
                        "\n"
                        "Record AMP as unsupported for this "
                        "configuration."
                    ) from exc

                raise

            loss.backward()

            optimizer.step()

            batch_size_actual = y.numel()

            seen += batch_size_actual

            running_loss += (
                loss.item() * batch_size_actual
            )

        # Accelerator operations can be asynchronous.
        # Synchronise before stopping the timer.
        sync_device(device)

        seconds = time.perf_counter() - start

        accuracy = evaluate(
            model,
            test_loader,
            device,
        )

        # Make sure evaluation GPU work has completed.
        sync_device(device)

        row = {
            "epoch": epoch,
            "loss": running_loss / seen,
            "val_accuracy": accuracy,
            "seconds": seconds,
            "images_per_second": seen / seconds,
            "accelerator_memory_mb":
                accelerator_memory_mb(device),
            "device": device.type,
            "batch_size": args.batch_size,
            "amp": args.amp,
        }

        history.append(row)

        print(json.dumps(row))

        scheduler.step()

    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    precision_name = (
        "amp"
        if args.amp
        else "fp32"
    )

    checkpoint_path = (
        out
        / (
            f"resnet18_"
            f"{args.device}_"
            f"bs{args.batch_size}_"
            f"{precision_name}.pt"
        )
    )

    metrics_path = (
        out
        / (
            f"metrics_"
            f"{args.device}_"
            f"bs{args.batch_size}_"
            f"{precision_name}.json"
        )
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "history": history,
            "args": vars(args),
        },
        checkpoint_path,
    )

    metrics_path.write_text(
        json.dumps(
            history,
            indent=2,
        )
    )

    print()
    print("Training complete.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics:    {metrics_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default="mps",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision on MPS or CUDA.",
    )

    args = parser.parse_args()

    train(args)