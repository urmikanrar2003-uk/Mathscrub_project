"""
Training script for ViT Strikeout Detector - Phase 2 with Real Data

This script trains the Vision Transformer model for strikeout detection
using actual component-level annotations from the dataset.
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
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import logging

from vit_strikeout_detector import ViTStrikeoutDetector, StrikeoutDetector
from data_loader_phase2 import DatasetBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrikeoutTrainer:
    """
    Trainer for strikeout detection model with real data.
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
            'train_precision': [],
            'train_recall': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rate': []
        }
        
        # Best metrics
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
    
    def setup_optimizer(self, learning_rate: float = 1e-4,
                       weight_decay: float = 1e-5) -> optim.Optimizer:
        """Setup optimizer with weight decay."""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        return optimizer
    
    def setup_scheduler(self, optimizer: optim.Optimizer,
                       epochs: int, warmup_epochs: int = 5):
        """Setup learning rate scheduler with warmup."""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / 
                                    (epochs - warmup_epochs)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
    
    def compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute precision and recall."""
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {'precision': precision, 'recall': recall}
    
    def train_epoch(self, train_loader: DataLoader,
                   optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Train one epoch.
        
        Returns:
            Dictionary with loss, accuracy, precision, recall
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
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
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        metrics = self.compute_metrics(all_preds, all_labels)
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total,
            'precision': metrics['precision'],
            'recall': metrics['recall']
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary with loss, accuracy, precision, recall
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits, y)
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        metrics = self.compute_metrics(all_preds, all_labels)
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total,
            'precision': metrics['precision'],
            'recall': metrics['recall']
        }
    
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
            logger.info(f"Saved best checkpoint: {best_path}")
        
        # Save epoch checkpoint every N epochs
        if (epoch + 1) % 10 == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch_{epoch+1:04d}.pt'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('epoch', 0)
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 30, learning_rate: float = 1e-4,
            weight_decay: float = 1e-5, warmup_epochs: int = 2):
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
        
        logger.info(f"Training on {self.device}")
        logger.info(f"Epochs: {epochs}, LR: {learning_rate}, Weight Decay: {weight_decay}")
        logger.info(f"Total samples: Train={len(train_loader.dataset)}, "
                   f"Val={len(val_loader.dataset)}")
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_precision'].append(train_metrics['precision'])
            self.history['train_recall'].append(train_metrics['recall'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            
            # Learning rate tracking
            current_lr = optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            print(f"\nTrain Metrics:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"  Precision: {train_metrics['precision']:.4f}")
            print(f"  Recall: {train_metrics['recall']:.4f}")
            ge_img
        
        if not img_path.exists():
            print(f"\nValidation Metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall: {val_metrics['recall']:.4f}")
            
            print(f"\nLearning Rate: {current_lr:.6f}")
            
            # Check best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                print(f"\n✓ New best validation accuracy: {val_metrics['accuracy']:.4f}")
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
            
            # Save checkpoint
            self.save_checkpoint(epoch, optimizer, is_best=is_best)
            
            # Scheduler sge_img
        
        if not img_path.exists():tep
            scheduler.step()
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("="*60)
        
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
        
        logger.info(f"Saved metrics to {metrics_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train ViT strikeout detector on real data"
    )
    
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
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                       help='Warmup epochs')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    # Data args
    parser.add_argument('--data-dir', type=str, 
                       default='./data',
                       help='Data directory path')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--use-synth', action='store_true', default=False,
                       help='Use synthetic data instead of real')
    
    # I/O args
    parser.add_argument('--output-dir', type=str, default='checkpoints_phase2',
                       help='Output directory for checkpoints')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize model
    logger.info("Initializing ViT Strikeout Detector...")
    model = ViTStrikeoutDetector(
        num_classes=2,
        pretrained=args.pretrained,
        image_size=args.image_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        num_layers=args.num_layers
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloaders
    logger.info("\nLoading datasets...")
    train_loader, val_loader = DatasetBuilder.create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_synth=args.use_synth
    )
    
    # Initialize trainer
    trainer = StrikeoutTrainer(model, output_dir=args.output_dir)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from)
        logger.info(f"Resuming from epoch {start_epoch + 1}")
    
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
