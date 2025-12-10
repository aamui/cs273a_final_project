"""
SVHN CNN Ensemble: train & eval in one file.

Usage:

  # Train an ensemble of 3 models (default)
  python cnn_ensemble.py --mode train --n_ensemble 3

  # Evaluate all checkpoints in out_dir (default: ./checkpoints)
  python cnn_ensemble.py --mode eval
  python cnn_ensemble.py --mode eval --tta --tta_n 4 (use TTA with 4 views)

  # Custom options
  python cnn_ensemble.py --mode train --epochs 20 --batch_size 128
  python cnn_ensemble.py --mode eval --out_dir ./checkpoints
"""

import argparse
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

# ---------------------------
# Model definition
# ---------------------------

class SmallSVHNCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------
# Data loading
# ---------------------------

def get_dataloaders(
    data_dir: str,
    batch_size: int,
    val_fraction: float = 0.1,
    num_workers: int = 4,
):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    full_train = datasets.SVHN(
        root=data_dir, split='train', download=True, transform=train_transform
    )
    test_dataset = datasets.SVHN(
        root=data_dir, split='test', download=True, transform=test_transform
    )

    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

# ---------------------------
# Augmentation helpers
# ---------------------------

def tta_augment_batch(x, n_aug: int):
    """
    Given a batch x (B, C, H, W), return a list of length n_aug
    of augmented batches. First one can just be the original.
    """
    batches = [x]  # always include original

    for _ in range(n_aug - 1):
        x_aug = []
        for img in x:
            # img: (C, H, W) tensor
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                img_t = F.hflip(img)
            else:
                img_t = img

            # Random crop with padding 4 (like train)
            img_t = F.pad(img_t, padding=4, padding_mode="reflect")
            img_t = F.crop(img_t,
                           top=torch.randint(0, 5, (1,)).item(),
                           left=torch.randint(0, 5, (1,)).item(),
                           height=32,
                           width=32)

            x_aug.append(img_t)
        x_aug = torch.stack(x_aug, dim=0)
        batches.append(x_aug)

    return batches  # list of (B, C, H, W) tensors

# ---------------------------
# Training / evaluation helpers
# ---------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        # SVHN returns labels as LongTensor already, but be explicit:
        y = y.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device, dtype=torch.long)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def train_single_model(
    run_id: int,
    train_loader,
    val_loader,
    device,
    out_dir: str,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay=5e-4,
):
    print(f"\n=== Training model run {run_id} ===")

    # basic seeding per run
    torch.manual_seed(42 + run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + run_id)

    model = SmallSVHNCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"svhn_cnn_run{run_id}.pt")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        print(
            f"[Run {run_id}] Epoch {epoch+1}/{epochs} "
            f"Train loss {train_loss:.3f} acc {train_acc:.3f} | "
            f"Val loss {val_loss:.3f} acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
        
        scheduler.step()

    print(f"[Run {run_id}] Best val acc: {best_val_acc:.4f} | Saved: {ckpt_path}")
    return ckpt_path, best_val_acc


@torch.no_grad()
def eval_single_checkpoint(path, test_loader, device):
    model = SmallSVHNCNN().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print(f"Single model [{os.path.basename(path)}] Test loss {test_loss:.4f} acc {test_acc:.4f}")
    return test_loss, test_acc


@torch.no_grad()
def eval_ensemble(ckpt_paths, test_loader, device, use_tta=False, tta_n=4):
    if len(ckpt_paths) == 0:
        print("No checkpoints found for ensemble evaluation.")
        return None

    print("\n=== Evaluating ensemble ===")
    if use_tta:
        print(f"Using TTA with {tta_n} views per image")

    models = []
    for path in ckpt_paths:
        m = SmallSVHNCNN().to(device)
        state = torch.load(path, map_location=device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
        print(f"Loaded: {path}")

    correct = 0
    total = 0

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device, dtype=torch.long)

        if use_tta:
            # list of length tta_n of (B, C, H, W) tensors
            tta_batches = tta_augment_batch(x, n_aug=tta_n)
        else:
            tta_batches = [x]

        logits_sum = None

        for m in models:
            for x_view in tta_batches:
                x_view = x_view.to(device)
                logits = m(x_view)
                if logits_sum is None:
                    logits_sum = logits
                else:
                    logits_sum += logits

        logits_mean = logits_sum / (len(models) * len(tta_batches))
        _, preds = logits_mean.max(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Ensemble (n={len(models)}, TTA={use_tta}) Test acc: {acc:.4f}")
    return acc


# ---------------------------
# Modes: train / eval
# ---------------------------

def run_train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
    )

    ckpt_paths = []
    val_accs = []

    for run_id in range(args.n_ensemble):
        path, val_acc = train_single_model(
            run_id=run_id,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            out_dir=args.out_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        ckpt_paths.append(path)
        val_accs.append(val_acc)

    print("\n=== Training complete ===")
    for p, a in zip(ckpt_paths, val_accs):
        print(f"{p} | best val acc {a:.4f}")

def run_eval(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
    )

    # Find checkpoints
    if args.ckpt_paths:
        ckpt_paths_all = [p.strip() for p in args.ckpt_paths.split(",")]
    else:
        pattern = os.path.join(args.out_dir, "svhn_cnn_run*.pt")
        ckpt_paths_all = sorted(glob.glob(pattern))

    if len(ckpt_paths_all) == 0:
        print(f"No checkpoints found. Looked for: {pattern}")
        return

    print("\n=== Evaluating individual models ===")
    for path in ckpt_paths_all:
        eval_single_checkpoint(path, test_loader, device)

    # Sweep M = 1..len(ckpt_paths_all)
    print("\n=== Ensemble sweep ===")
    for M in range(1, len(ckpt_paths_all) + 1):
        ckpts_subset = ckpt_paths_all[:M]
        print(f"\nEvaluating Ensemble M = {M}")
        eval_ensemble(
            ckpt_paths=ckpts_subset,
            test_loader=test_loader,
            device=device,
            use_tta=args.tta,
            tta_n=args.tta_n,
        )

# ---------------------------
# Argument parsing
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SVHN CNN Ensemble (train/eval)")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        required=True,
        help="train: train ensemble; eval: evaluate saved models",
    )

    # common args
    parser.add_argument("--data_dir", type=str, default="./data", help="SVHN data directory")
    parser.add_argument("--out_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    # train-specific
    parser.add_argument("--n_ensemble", type=int, default=5, help="Number of models in ensemble")
    parser.add_argument("--epochs", type=int, default=60, help="Epochs per model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for optimizer")

    # eval-specific
    parser.add_argument(
        "--ckpt_paths",
        type=str,
        default="",
        help="Comma-separated list of checkpoint paths (optional). "
             "If empty, will glob svhn_cnn_run*.pt in out_dir.",
    )

    # test-time augmentation (TTA)
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use test-time augmentation during evaluation",
    )
    parser.add_argument(
        "--tta_n",
        type=int,
        default=4,
        help="Number of TTA views per image (including original) when --tta is enabled",
    )

    return parser.parse_args()


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)

