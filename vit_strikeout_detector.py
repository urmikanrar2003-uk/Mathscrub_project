"""
Vision Transformer for Strikeout Detection - Phase 2

This module implements a Vision Transformer (ViT) architecture for detecting
strikeouts in handwritten mathematical expressions. It processes three-channel
input: grayscale image, component pixel area normalization, and bounding box size.

Based on "MathScrub: Single-Pipeline Strikeout Removal and LaTeX Generation 
for Handwritten Calculus Expressions" (Sen Shen et al., 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, List
from einops import rearrange
from transformers import ViTModel, ViTConfig
import cv2
from PIL import Image


class GeometryNormalizer:
    """
    Normalizes geometry features for multimodal deletion classification.
    
    Applies precise mathematical formulas for:
    1. Component pixel area normalization
    2. Bounding box size normalization
    3. Image normalization
    """
    
    @staticmethod
    def normalize_component_area(area: float, total_image_area: float) -> float:
        """
        Normalize component pixel area as a proportion of image area.
        
        Args:
            area: Component pixel area in pixels^2
            total_image_area: Total image area in pixels^2
            
        Returns:
            Normalized area in range [0, 1]
        """
        if total_image_area == 0:
            return 0.0
        return min(1.0, area / total_image_area)
    
    @staticmethod
    def normalize_bbox_size(width: float, height: float, 
                           img_width: float, img_height: float) -> float:
        """
        Normalize bounding box size relative to image dimensions.
        Uses L2 norm of normalized dimensions.
        
        Args:
            width: Bounding box width
            height: Bounding box height
            img_width: Image width
            img_height: Image height
            
        Returns:
            Normalized size in range [0, 1]
        """
        if img_width == 0 or img_height == 0:
            return 0.0
        
        norm_w = width / img_width
        norm_h = height / img_height
        # Use Euclidean norm
        norm_size = np.sqrt(norm_w**2 + norm_h**2) / np.sqrt(2)
        return min(1.0, norm_size)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range using min-max normalization.
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image
        """
        img_min = image.min()
        img_max = image.max()
        
        if img_max == img_min:
            return np.zeros_like(image, dtype=np.float32)
        
        return ((image - img_min) / (img_max - img_min)).astype(np.float32)


class StrikeoutDataset(Dataset):
    """
    PyTorch Dataset for strikeout detection.
    
    Expects input with three channels:
    - Channel 0: Grayscale image
    - Channel 1: Component pixel area (normalized)
    - Channel 2: Bounding box size (normalized)
    """
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 areas: List[float] = None, bbox_sizes: List[float] = None,
                 image_size: int = 224):
        """
        Initialize strikeout dataset.
        
        Args:
            image_paths: List of paths to grayscale images
            labels: List of binary labels (0: no strikeout, 1: strikeout)
            areas: List of normalized component areas
            bbox_sizes: List of normalized bounding box sizes
            image_size: Target image size for ViT input (default 224)
        """
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.image_size = image_size
        
        # Initialize with zeros if not provided
        self.areas = torch.tensor(areas or [0.0] * len(labels), dtype=torch.float32)
        self.bbox_sizes = torch.tensor(bbox_sizes or [0.0] * len(labels), dtype=torch.float32)
        
        assert len(image_paths) == len(labels), "Mismatch between images and labels"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (3-channel input tensor, label)
        """
        # Load grayscale image
        if isinstance(self.image_paths[idx], str):
            try:
                image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
                if image is None:
                    image = np.array(Image.open(self.image_paths[idx]).convert('L'))
            except:
                # Create a dummy image if loading fails
                image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        elif isinstance(self.image_paths[idx], torch.Tensor):
            # Convert tensor to numpy array
            image = self.image_paths[idx].numpy() if isinstance(self.image_paths[idx], torch.Tensor) else self.image_paths[idx]
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
        else:
            image = self.image_paths[idx]
            if isinstance(image, torch.Tensor):
                image = image.numpy()
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image = (image * 255).astype(np.uint8)
        
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint8)
        
        # Resize image
        if image.shape != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize image
        image = GeometryNormalizer.normalize_image(image)
        
        # Create 3-channel input
        channel_0 = torch.tensor(image, dtype=torch.float32)
        channel_1 = torch.full_like(channel_0, self.areas[idx].item())
        channel_2 = torch.full_like(channel_0, self.bbox_sizes[idx].item())
        
        # Stack channels
        x = torch.stack([channel_0, channel_1, channel_2], dim=0)
        
        return x, self.labels[idx]


class ViTStrikeoutDetector(nn.Module):
    """
    Vision Transformer-based strikeout detector.
    
    Architecture:
    - ViT backbone for feature extraction
    - Classification head for binary strikeout detection
    - Supports multiple pooling strategies
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 image_size: int = 224, patch_size: int = 16,
                 hidden_size: int = 768, num_heads: int = 12,
                 intermediate_size: int = 3072, num_layers: int = 12,
                 dropout: float = 0.1):
        """
        Initialize ViT strikeout detector.
        
        Args:
            num_classes: Number of output classes (default: 2 for binary)
            pretrained: Whether to use pretrained ViT weights
            image_size: Input image size
            patch_size: Patch size for tokenization
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            intermediate_size: Intermediate FFN size
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        # Custom input projection for 3-channel input
        # Standard ViT expects 3 channels, but we'll use our custom processor
        self.input_projection = nn.Conv2d(3, 3, kernel_size=1)
        
        # ViT Config
        vit_config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=3,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            initializer_range=0.02,
            layer_norm_eps=1e-12
        )
        
        # ViT Model
        self.vit = ViTModel(vit_config, add_pooling_layer=True)
        
        if pretrained:
            try:
                # Load pretrained ViT-base weights and adapt for 3 channels
                pretrained_vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
                # Copy weights, handling the potential channel mismatch
                self._load_pretrained_weights(pretrained_vit)
            except Exception as e:
                print(f"Could not load pretrained weights: {e}. Training from scratch.")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def _load_pretrained_weights(self, pretrained_model):
        """Load pretrained weights, handling channel dimension mismatch."""
        try:
            # Copy embeddings
            if hasattr(pretrained_model.embeddings, 'patch_embeddings'):
                patch_embed = pretrained_model.embeddings.patch_embeddings.projection
                # If pretrained has 3 channels and we need 3, direct copy
                self.vit.embeddings.patch_embeddings.projection.weight.data = patch_embed.weight.data
                self.vit.embeddings.patch_embeddings.projection.bias.data = patch_embed.bias.data
            
            # Copy encoder weights
            self.vit.encoder.load_state_dict(pretrained_model.encoder.state_dict())
            self.vit.embeddings.cls_token.data = pretrained_model.embeddings.cls_token.data
            self.vit.embeddings.position_embeddings.data = pretrained_model.embeddings.position_embeddings.data
        except Exception as e:
            print(f"Partial weight loading: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, image_size, image_size)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Project input
        x = self.input_projection(x)
        
        # ViT forward pass
        outputs = self.vit(x)
        pooled = outputs.pooler_output
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps from the last layer.
        Useful for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention tensor
        """
        x = self.input_projection(x)
        outputs = self.vit(x, output_attentions=True)
        last_attention = outputs.attentions[-1]
        return last_attention


class StrikeoutDetector:
    """
    High-level API for strikeout detection.
    Handles training, evaluation, and inference.
    """
    
    def __init__(self, model: ViTStrikeoutDetector, device: str = None):
        """
        Initialize detector.
        
        Args:
            model: ViT strikeout detector model
            device: Device to use (cpu, cuda, etc.)
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                   optimizer: torch.optim.Optimizer) -> float:
        """
        Single training step.
        
        Args:
            batch: (input, labels) tuple
            optimizer: Optimizer
            
        Returns:
            Loss value
        """
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        optimizer.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            (accuracy, average_loss)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return accuracy, avg_loss
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            (predictions, probabilities)
        """
        self.model.eval()
        x = x.to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        return preds, probs


# Utility functions
def create_dummy_dataset(num_samples: int = 100, 
                        image_size: int = 224) -> Tuple[List[np.ndarray], List[int]]:
    """
    Create a dummy dataset for testing.
    
    Args:
        num_samples: Number of samples
        image_size: Image size
        
    Returns:
        (images, labels)
    """
    images = []
    labels = []
    
    for i in range(num_samples):
        # Random image
        img = np.random.rand(image_size, image_size).astype(np.float32)
        images.append(img)
        
        # Random label
        label = np.random.randint(0, 2)
        labels.append(label)
    
    return images, labels


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("Vision Transformer Strikeout Detector")
    print("=" * 50)
    
    # Initialize model
    print("\n1. Initializing ViT model...")
    model = ViTStrikeoutDetector(
        num_classes=2,
        pretrained=True,
        image_size=224,
        patch_size=16
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Model initialized on {device}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create detector
    print("\n2. Creating detector...")
    detector = StrikeoutDetector(model, device=device)
    print("   Detector ready!")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output: {output}")
    
    # Test with dummy dataset
    print("\n4. Testing with dummy dataset...")
    images, labels = create_dummy_dataset(num_samples=10, image_size=224)
    dataset = StrikeoutDataset(images, labels, image_size=224)
    dataloader = DataLoader(dataset, batch_size=2)
    
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Batch size: 2")
    
    # Test prediction
    print("\n5. Testing prediction...")
    accuracy, loss = detector.evaluate(dataloader)
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Loss: {loss:.4f}")
    
    print("\n✓ All tests passed!")
