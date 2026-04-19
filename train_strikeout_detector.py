"""
Training script for ViT Strikeout Detector - Phase 2

This script trains the Vision Transformer model for strikeout detection
with support for checkpointing, validation, and metrics tracking.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

from vit_strikeout_detector import (
    ViTStrikeoutDetector,
    StrikeoutDetector,
    StrikeoutDataset,
    GeometryNormalizer
)


class StrikeoutTrainer:
    """
    Trainer for strikeout detection model.
    Handles training loop, validation, checkpointing, and metrics.
    """
    
    def __init__(self, model: ViTStrikeoutDetector, device: str = None,
                 output_dir: str = "checkpoints"):
        """
        Initialize trainer.
        
        Args:
            model: ViT model
            device: Device to use
            output_dir: Directory for checkpoints
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.detector = StrikeoutDetector(model, device=self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Best metrics
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
    
    def setup_optimizer(self, learning_rate: float = 1e-3,
                       weight_decay: float = 1e-4) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        return optimizer
    
    def setup_scheduler(self, optimizer: optim.Optimizer,
                       epochs: int, warmup_epochs: int = 5):
        """Setup learning rate scheduler."""
        total_steps = epochs
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
    
    def train_epoch(self, train_loader: DataLoader,
                   optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        Train one epoch.
        
        Returns:
            (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            pbar.set_postfix({'loss': loss.item():.4f})
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Returns:
            (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits, y)
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, 
                       is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
        
        # Save epoch checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch_{epoch+1:03d}.pt'
            torch.save(checkpoint, epoch_path)
        
        return latest_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        return checkpoint.get('epoch', 0)
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 50, learning_rate: float = 1e-3,
            weight_decay: float = 1e-4, warmup_epochs: int = 5):
        """
        Full training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Number of epochs
            learning_rate: Initial learning rate
            weight_decay: Weight decay
            warmup_epochs: Warmup epochs
        """
        optimizer = self.setup_optimizer(learning_rate, weight_decay)
        scheduler = self.setup_scheduler(optimizer, epochs, warmup_epochs)
        
        print(f"Training on {self.device}")
        print(f"Epochs: {epochs}, LR: {learning_rate}, Weight Decay: {weight_decay}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate tracking
            current_lr = optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Check best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f"✓ New best validation accuracy: {val_acc:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, optimizer, is_best=is_best)
            
            # Scheduler step
            scheduler.step()
        
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        
        # Save final metrics
        self.save_metrics()
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics_path = self.output_dir / 'metrics.json'
        metrics = {
            'history': {k: v for k, v in self.history.items()},
            'best_val_acc': float(self.best_val_acc),
            'best_val_loss': float(self.best_val_loss)
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Saved metrics to {metrics_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train ViT strikeout detector")
    
    # Model args
    parser.add_argument('--hidden-size', type=int, default=768,
                       help='ViT hidden size')
    parser.add_argument('--num-heads', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    # Data args
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--train-size', type=int, default=800,
                       help='Training set size')
    parser.add_argument('--val-size', type=int, default=200,
                       help='Validation set size')
    
    # I/O args
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize model
    print("Initializing ViT Strikeout Detector...")
    model = ViTStrikeoutDetector(
        num_classes=2,
        pretrained=args.pretrained,
        image_size=args.image_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        num_layers=args.num_layers
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy datasets for testing
    print("\nCreating dummy datasets...")
    from vit_strikeout_detector import create_dummy_dataset
    
    train_images, train_labels = create_dummy_dataset(
        num_samples=args.train_size,
        image_size=args.image_size
    )
    
    val_images, val_labels = create_dummy_dataset(
        num_samples=args.val_size,
        image_size=args.image_size
    )
    
    # Create datasets and dataloaders
    train_dataset = StrikeoutDataset(train_images, train_labels, image_size=args.image_size)
    val_dataset = StrikeoutDataset(val_images, val_labels, image_size=args.image_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train set: {len(train_dataset)}")
    print(f"Val set: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = StrikeoutTrainer(model, output_dir=args.output_dir)
    
    # Train
    trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs
    )


if __name__ == '__main__':
    main()
