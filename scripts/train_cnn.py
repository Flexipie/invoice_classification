"""
Train a custom CNN for document image classification.

Architecture (Model C):
  Conv(1→32, 3×3) → BN → ReLU → MaxPool(2)
  Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)
  Conv(64→128, 3×3) → BN → ReLU → MaxPool(2)
  Flatten → FC(32768→512) → Dropout(0.5) → FC(512→4)

Input: 128×128 grayscale images (1 channel).
Best checkpoint (highest val accuracy) saved to models/cnn_best.pth.
Training history saved to reports/cnn_training.json.

Override defaults via env vars:
  EPOCHS      (default 20)
  BATCH_SIZE  (default 32)
  LR          (default 1e-3)
  DEVICE      (default: auto-detect cuda/mps/cpu)
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPOCHS     = int(os.environ.get("EPOCHS", 20))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
LR         = float(os.environ.get("LR", 1e-3))

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

NUM_CLASSES = 4
IMG_SIZE    = 128


def get_device() -> torch.device:
    name = os.environ.get("DEVICE", "")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DocumentDataset(Dataset):
    MEAN = 0.5
    STD  = 0.5

    def __init__(self, split: str, augment: bool = False):
        self.images = np.load(DATA_DIR / f"{split}_images.npy")   # (N, H, W) uint8
        self.labels = np.load(DATA_DIR / f"{split}_labels.npy")   # (N,)
        self.augment = augment

        base = [
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.MEAN], std=[self.STD]),
        ]
        if augment:
            aug = [
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.MEAN], std=[self.STD]),
            ]
            self.transform = transforms.Compose(aug)
        else:
            self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]   # (H, W) uint8
        return self.transform(img), int(self.labels[idx])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DocumentCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1×128×128 → 32×64×64
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32×64×64 → 64×32×32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64×32×32 → 128×16×16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # 128 × 16 × 16 = 32768
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(lbls)
        correct    += (out.argmax(1) == lbls).sum().item()
        n          += len(lbls)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model(imgs)
        loss = criterion(out, lbls)
        total_loss += loss.item() * len(lbls)
        correct    += (out.argmax(1) == lbls).sum().item()
        n          += len(lbls)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"=== CNN Training  (device={device}, epochs={EPOCHS}, bs={BATCH_SIZE}, lr={LR}) ===\n")

    train_ds = DocumentDataset("train",      augment=True)
    val_ds   = DocumentDataset("validation", augment=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = DocumentCNN(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    best_val_acc = 0.0
    best_path    = MODELS_DIR / "cnn_best.pth"

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model,  val_loader,   criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc":  round(tr_acc, 4),
            "val_loss":   round(vl_loss, 4),
            "val_acc":    round(vl_acc, 4),
        })

        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": vl_acc,
                "num_classes": NUM_CLASSES,
            }, best_path)
            flag = " ← best"

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val loss={vl_loss:.4f} acc={vl_acc:.4f}{flag}"
        )

    with open(REPORTS_DIR / "cnn_training.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved → {best_path}")
    print("Next step: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
