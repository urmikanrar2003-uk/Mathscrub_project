"""
train_vit.py
============
Trains the MultimodalDeletionViT using labeled component crops
from the MathStrike 'real' and 'synth' HuggingFace dataset splits.

Each sample in these splits contains:
    image      — cropped component image (PIL)
    label      — 0 (retain/math) or 1 (delete/strikeout)
    box_area   — pixel area of the component  → X_area geometry scalar
    comp_size  — [height, width] of bbox       → X_size geometry scalar

Training strategy:
    - Streams 'real' + 'synth' splits
    - Constructs 3-channel tensors (Image + Area + Size)
    - Trains with BCEWithLogitsLoss + AdamW optimizer
    - Saves best model checkpoint as best_model.pth
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from vit_strikeout_detector import MultimodalDeletionViT, pad_to_square

load_dotenv()
hf_token = os.getenv("HF_token")

# ── Dataset Statistics (from stats_for_VIT.ipynb) ───────────────────────────
MU_A    = 8.6510
SIGMA_A = 0.9201
MU_S    = 5.1047
SIGMA_S = 0.6324


# ═══════════════════════════════════════════════════════════════════════════
#  PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════

class MathStrikeDeletionDataset(Dataset):
    """
    Converts MathStrike 'real' / 'synth' HuggingFace samples into
    3-channel (Image + Area + Size) ViT input tensors with binary labels.
    """
    def __init__(self, hf_samples, target_size=224):
        self.samples = list(hf_samples)
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ── Channel 0: Grayscale image crop ──────────────────────────────
        pil_img = sample["image"]
        if pil_img.mode != "L":
            pil_img = pil_img.convert("L")
        img_np = np.array(pil_img, dtype=np.uint8)
        resized = pad_to_square(img_np, target_size=self.target_size, padding_value=255)
        img_ch  = resized.astype(np.float32) / 255.0

        # ── Channel 1: X_area scalar broadcast ───────────────────────────
        A = sample["box_area"]
        x_area = (math.log(A + 1) - MU_A) / SIGMA_A
        area_ch = np.full((self.target_size, self.target_size), x_area, dtype=np.float32)

        # ── Channel 2: X_size scalar broadcast ───────────────────────────
        # comp_size = [height, width]; L = max(w, h)
        comp_size = sample["comp_size"]
        L = max(comp_size[0], comp_size[1])
        x_size = (math.log(L + 1) - MU_S) / SIGMA_S
        size_ch = np.full((self.target_size, self.target_size), x_size, dtype=np.float32)

        # Stack → (3, H, W) tensor
        tensor = torch.from_numpy(np.stack([img_ch, area_ch, size_ch], axis=0))
        label  = torch.tensor(float(sample["label"]), dtype=torch.float32)

        return tensor, label


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_vit(
    dataset_name = "Incinciblecolonel/MathStrike",
    train_limit  = 2000,     # samples from real + synth combined (training)
    val_limit    = 500,      # samples from real test split (validation)
    epochs       = 10,
    batch_size   = 16,
    lr           = 1e-4,
    checkpoint   = "best_model.pth",
    pretrained   = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # ── 1. Load TRAINING data: real(train) + synth(train) ────────────────
    print("Loading real/train split...")
    ds_real_train  = load_dataset(dataset_name, data_dir="real",  split="train",
                                  streaming=True, token=hf_token)
    print("Loading synth/train split...")
    ds_synth_train = load_dataset(dataset_name, data_dir="synth", split="train",
                                  streaming=True, token=hf_token)

    print(f"Collecting training samples (limit={train_limit})...")
    train_samples = []
    half = (train_limit // 2) if train_limit is not None else None
    for i, s in enumerate(ds_real_train):
        if half is not None and i >= half: break
        train_samples.append(s)
    for i, s in enumerate(ds_synth_train):
        if half is not None and i >= half: break
        train_samples.append(s)
    np.random.shuffle(train_samples)
    print(f"Training samples: {len(train_samples)} "
          f"(label=1: {sum(s['label']==1 for s in train_samples)}, "
          f"label=0: {sum(s['label']==0 for s in train_samples)})")

    # ── 2. Load VALIDATION data: real(test) — native HuggingFace test split ──
    print("Loading real/test split for validation...")
    ds_real_test = load_dataset(dataset_name, data_dir="real", split="test",
                                streaming=True, token=hf_token)
    val_samples = []
    for i, s in enumerate(ds_real_test):
        if val_limit is not None and i >= val_limit: break
        val_samples.append(s)
    print(f"Validation samples: {len(val_samples)} "
          f"(label=1: {sum(s['label']==1 for s in val_samples)}, "
          f"label=0: {sum(s['label']==0 for s in val_samples)})")

    train_ds = MathStrikeDeletionDataset(train_samples)
    val_ds   = MathStrikeDeletionDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── 3. Model, Loss, Optimizer ─────────────────────────────────────────
    model = MultimodalDeletionViT(pretrained=pretrained).to(device)

    # BCEWithLogitsLoss expects raw logits (before sigmoid) → we need to
    # temporarily bypass the sigmoid in the network during training.
    # We add a raw logit output path by wrapping around the trained head:
    criterion = nn.BCELoss()         # model already applies sigmoid
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")

    # ── 4. Epoch loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0

        for tensors, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            tensors = tensors.to(device)
            labels  = labels.to(device)

            optimizer.zero_grad()
            probs = model(tensors)           # (B,) probabilities from sigmoid
            loss  = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * tensors.size(0)
            preds          = (probs >= 0.5).float()
            train_correct += (preds == labels).sum().item()

        train_loss     /= len(train_ds)
        train_acc       = train_correct / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for tensors, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]  "):
                tensors = tensors.to(device)
                labels  = labels.to(device)
                probs   = model(tensors)
                loss    = criterion(probs, labels)
                val_loss    += loss.item() * tensors.size(0)
                preds        = (probs >= 0.5).float()
                val_correct += (preds == labels).sum().item()

        val_loss    /= len(val_ds)
        val_acc      = val_correct / len(val_ds)

        scheduler.step()

        print(f"Epoch {epoch:>2}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint)
            print(f"  ✓ Saved best model → {checkpoint}")

    print(f"\nTraining complete! Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoint saved at: {checkpoint}")
    return model, checkpoint


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_vit(
        train_limit = None,    # None = full real+synth dataset (recommended on GPU)
                               # Set e.g. 200 for a quick local sanity check
        val_limit   = None,    # None = full real/test split
        epochs      = 10,
        batch_size  = 32,      # increase to 32 on GPU (was 16 for CPU)
        lr          = 1e-4,
    )
