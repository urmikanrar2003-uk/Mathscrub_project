"""
vit_strikeout_detector.py
=========================
Defines the Early-Fusion Vision Transformer (ViT) for Multimodal Deletion Classification.

Architecture (MathScrub Section B):
    Input  : 3-channel tensor (224×224)
               Channel 0 — Grayscale crop of component (normalized to [0,1])
               Channel 1 — X_area scalar broadcast  = (log(A+1) - mu_a) / sigma_a
               Channel 2 — X_size scalar broadcast  = (log(L+1) - mu_s) / sigma_s
    Encoder: torchvision ViT_B_16
    Head   : Linear(hidden_dim, 1)  →  Sigmoid  →  p_delete ∈ [0,1]

Usage:
    from vit_strikeout_detector import MultimodalDeletionViT, pad_to_square

Training:
    See train_vit.py

Inference (after training):
    model = MultimodalDeletionViT()
    model.load_state_dict(torch.load('best_model.pth'))
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

# ── Dataset Statistics (from stats_for_VIT.ipynb — real + synth full dataset) ──
MU_A    = 8.6510
SIGMA_A = 0.9201
MU_S    = 5.1047
SIGMA_S = 0.6324


# ═══════════════════════════════════════════════════════════════════════════
#  PREPROCESSING UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def pad_to_square(img_crop, target_size=224, padding_value=255):
    """
    Pads a grayscale crop to make it square (preserving aspect ratio),
    then resizes to target_size × target_size.

    Args:
        img_crop     : 2D numpy array (grayscale, uint8)
        target_size  : output resolution (default 224 for ViT)
        padding_value: fill value for padding (255 = white background)

    Returns:
        resized 2D numpy array of shape (target_size, target_size)
    """
    h, w = img_crop.shape[:2]
    max_dim = max(h, w)

    pad_h = max_dim - h
    pad_w = max_dim - w
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left

    squared = cv2.copyMakeBorder(
        img_crop, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=padding_value
    )
    return cv2.resize(squared, (target_size, target_size), interpolation=cv2.INTER_AREA)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════

class MultimodalDeletionViT(nn.Module):
    """
    Early-Fusion Vision Transformer for binary deletion classification.

    Wraps torchvision's ViT_B_16. The existing 3-channel patch-embedding
    naturally accepts our (Image, Area, Size) tensor. The default 1000-class
    head is replaced with a single-neuron sigmoid head that outputs p_delete.

    Args:
        pretrained: if True, initialise with ImageNet weights (useful as
                    a topological starting point before fine-tuning).
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)

        # Replace classification head: hidden_dim → 1 → Sigmoid
        hidden_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Linear(hidden_dim, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, 224, 224) float tensor
        Returns:
            probs : (B,) deletion probabilities ∈ [0, 1]
        """
        logits = self.vit(x)           # (B, 1)
        return self.sigmoid(logits).squeeze(-1)   # (B,)
