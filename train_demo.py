"""
Quick training demo for Phase 2 with real data.

This script demonstrates training the strikeout detector on real data
with limited samples for rapid iteration.
"""

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import logging

from vit_strikeout_detector import ViTStrikeoutDetector
from data_loader_phase2 import DatasetBuilder
from train_phase2_real_data import StrikeoutTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run quick demo training."""
    parser = argparse.ArgumentParser(description="Quick training demo")
    parser.add_argument('--data-dir', type=str, 
                       default='./data',
                       help='Data directory')
    parser.add_argument('--num-train-samples', type=int, default=5000,
                       help='Number of training samples to use')
    parser.add_argument('--num-test-samples', type=int, default=500,
                       help='Number of test samples to use')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='checkpoints_demo',
                       help='Output directory')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("ViT Strikeout Detector - Phase 2 Training Demo")
    logger.info("="*60)
    
    # Load full datasets
    logger.info("\nLoading datasets...")
    train_loader_full, test_loader_full = DatasetBuilder.create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        image_size=224,
        use_synth=False
    )
    
    # Create subsets for demo
    logger.info(f"\nCreating demo subsets:")
    logger.info(f"  Using {args.num_train_samples} training samples")
    logger.info(f"  Using {args.num_test_samples} test samples")
    
    train_dataset = train_loader_full.dataset
    test_dataset = test_loader_full.dataset
    
    # Create subsets
    train_subset = Subset(train_dataset, range(min(args.num_train_samples, len(train_dataset))))
    test_subset = Subset(test_dataset, range(min(args.num_test_samples, len(test_dataset))))
    
    # Create new dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # Initialize model
    logger.info("\nInitializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = ViTStrikeoutDetector(
        num_classes=2,
        pretrained=True,
        image_size=224,
        hidden_size=768,
        num_heads=12,
        num_layers=12
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = StrikeoutTrainer(model, device=device, output_dir=args.output_dir)
    
    # Train
    logger.info("\nStarting training...")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info("="*60)
    
    trainer.fit(
        train_loader,
        test_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=1e-5,
        warmup_epochs=1
    )
    
    logger.info("\n" + "="*60)
    logger.info("Demo training completed!")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
