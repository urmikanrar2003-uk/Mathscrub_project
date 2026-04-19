"""
Configuration file for Strikeout Detector project.

This file centralizes all configuration parameters and paths.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
REAL_DATA_DIR = DATA_DIR / 'real'
SYNTH_DATA_DIR = DATA_DIR / 'synth'
ORIGINAL_IMG_DIR = DATA_DIR / 'original_img'

# Data files
REAL_COMPONENT_JSON = REAL_DATA_DIR / 'component.json'
REAL_TRAIN_SPLIT = REAL_DATA_DIR / 'train_split.json'
REAL_TEST_SPLIT = REAL_DATA_DIR / 'test_split.json'
REAL_IMG_DIR = REAL_DATA_DIR / 'img'

SYNTH_COMPONENT_JSON = SYNTH_DATA_DIR / 'component.json'
SYNTH_TRAIN_SPLIT = SYNTH_DATA_DIR / 'train_split.json'
SYNTH_TEST_SPLIT = SYNTH_DATA_DIR / 'test_split.json'
SYNTH_IMG_DIR = SYNTH_DATA_DIR / 'img'

# Checkpoint directories
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
CHECKPOINT_DIR_PHASE2 = PROJECT_ROOT / 'checkpoints_phase2'
CHECKPOINT_DIR_DEMO = PROJECT_ROOT / 'checkpoints_demo'

# Model parameters
MODEL_DEFAULTS = {
    'num_classes': 2,
    'pretrained': True,
    'image_size': 224,
    'hidden_size': 768,
    'num_heads': 12,
    'num_layers': 12,
}

# Training parameters
TRAINING_DEFAULTS = {
    'batch_size': 64,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 30,
    'warmup_epochs': 5,
    'num_workers': 4,
}

# Device
DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'


def validate_data_structure():
    """
    Validate that all required data files and directories exist.
    
    Returns:
        bool: True if all data files exist, False otherwise
    """
    required_files = [
        REAL_COMPONENT_JSON,
        REAL_TRAIN_SPLIT,
        REAL_TEST_SPLIT,
        SYNTH_COMPONENT_JSON,
        SYNTH_TRAIN_SPLIT,
        SYNTH_TEST_SPLIT,
    ]
    
    required_dirs = [
        REAL_IMG_DIR,
        SYNTH_IMG_DIR,
        ORIGINAL_IMG_DIR,
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    missing_dirs = [d for d in required_dirs if not d.exists()]
    
    if missing_files:
        print("❌ Missing data files:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    if missing_dirs:
        print("❌ Missing data directories:")
        for d in missing_dirs:
            print(f"   - {d}")
        return False
    
    print("✅ Data structure is valid!")
    return True


def get_data_dir(use_synth: bool = False) -> Path:
    """
    Get the appropriate data directory based on the flag.
    
    Args:
        use_synth: If True, return synth data dir; else real data dir
    
    Returns:
        Path to the data directory
    """
    return SYNTH_DATA_DIR if use_synth else REAL_DATA_DIR


if __name__ == '__main__':
    print("=" * 60)
    print("Strikeout Detector - Configuration")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"\nValidating data structure...")
    validate_data_structure()
