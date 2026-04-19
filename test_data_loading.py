#!/usr/bin/env python
"""Quick test to verify data loading works."""

from data_loader_phase2 import DatasetBuilder

print("Testing data loading with local data...")
try:
    loaders = DatasetBuilder.create_dataloaders(
        './data', 
        batch_size=8, 
        num_workers=0
    )
    train_loader, test_loader = loaders
    print("✅ Data loading successful!")
    print(f"   Train loader: {len(train_loader)} batches")
    print(f"   Test loader: {len(test_loader)} batches")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
