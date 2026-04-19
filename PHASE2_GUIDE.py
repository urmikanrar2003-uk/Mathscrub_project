"""
PHASE 2: STRIKEOUT DETECTION - COMPLETE GUIDE

This guide explains how to work with the real dataset for Phase 2.
"""

# =============================================================================
# DATA STRUCTURE
# =============================================================================

"""
Your dataset is organized as follows:

./data/
├── real/
│   ├── component.json          # Full annotations (97,223 samples)
│   ├── train_split.json        # Training set annotations (97,223 samples)
│   ├── test_split.json         # Test set annotations (1,000 samples)
│   └── img/                    # [Not used in current pipeline]
│
├── synth/
│   ├── component.json          # Synthetic data annotations
│   ├── train_split.json
│   ├── test_split.json
│   └── img/
│
└── original_img/
    └── *.png                   # Full page images referenced in JSON

Data Statistics:
- Training: 97,223 components
  * No strikeout (label=0): 72,657
  * Strikeout (label=1):    24,566
  
- Testing: 1,000 components (balanced)
  * No strikeout (label=0): 500
  * Strikeout (label=1):    500

Each component has:
- page_img: Reference to full page image
- bbox: [x, y, height, width] bounding box in original image
- box_area: Bounding box area
- comp_size: [height, width] of component
- label: 0 (no strikeout) or 1 (strikeout)
"""

# =============================================================================
# QUICK START - DEMO TRAINING
# =============================================================================

"""
# 1. Test data loading
$ cd /home/ryukr2/Mathscrub_project
$ source .venv/bin/activate
$ python data_loader_phase2.py

# Expected output:
# - Loaded 97,223 training samples
# - Created 3039 training batches
# - Loaded 1,000 test samples
# - Created 32 test batches

# 2. Run quick demo (5K samples, 5 epochs)
$ python train_demo.py --epochs 5 --num-train-samples 5000

# 3. Monitor training
# - Checkpoints saved to: checkpoints_demo/
# - Best model: checkpoints_demo/checkpoint_best.pt
# - Metrics: checkpoints_demo/metrics.json
"""

# =============================================================================
# FULL TRAINING
# =============================================================================

"""
# Train on full dataset (97K samples, recommended: 30 epochs)
$ python train_phase2_real_data.py \\
    --data-dir ./data \\
    --epochs 30 \\
    --batch-size 64 \\
    --learning-rate 1e-4 \\
    --num-workers 4 \\
    --output-dir checkpoints_phase2_full

# Key hyperparameters:
- Learning rate: 1e-4 (use 1e-4 for real data, smaller than dummy data)
- Batch size: 64 (adjust based on GPU memory)
- Weight decay: 1e-5 (prevent overfitting)
- Warmup epochs: 2 (linear warmup)
- Scheduler: Cosine annealing with warmup

# Expected training time:
- ~2-4 hours on single GPU (batch size 64)
- RTX 3090: ~8-12 hours, RTX 4090: ~3-5 hours
"""

# =============================================================================
# INFERENCE ON NEW DATA
# =============================================================================

"""
# Single image
$ python inference_strikeout_detector.py \\
    --checkpoint checkpoints_phase2_full/checkpoint_best.pt \\
    --image /path/to/component.png

# Batch prediction
$ python inference_strikeout_detector.py \\
    --checkpoint checkpoints_phase2_full/checkpoint_best.pt \\
    --image-dir /path/to/components/
"""

# =============================================================================
# IMPORTANT NOTES
# =============================================================================

"""
1. DATA LOADING
   - Components are extracted from full page images on-the-fly
   - Bounding box is [x, y, height, width]
   - Images are normalized to [0, 1] using min-max scaling
   - 3-channel input: [grayscale, normalized_area, normalized_size]

2. GEOMETRY NORMALIZATION
   - Channel 1: Component area normalized by page area
   - Channel 2: Component size normalized by image diagonal
   - These features help the model understand component context

3. IMAGE CACHING
   - Full page images are cached during training for efficiency
   - First epoch may be slower due to image loading
   - Subsequent epochs use cached images

4. CLASS IMBALANCE
   - Training set: ~75% no strikeout, ~25% strikeout
   - Test set: 50% each (balanced)
   - Consider using class weights if needed

5. BEST PRACTICES
   - Use data augmentation for better generalization
   - Monitor both accuracy and per-class metrics
   - Save best model based on validation accuracy
   - Use learning rate scheduling with warmup
   - Consider mixed precision training for faster iterations

6. RESUMING TRAINING
   $ python train_phase2_real_data.py \\
       --resume-from checkpoints_phase2_full/checkpoint_latest.pt \\
       --epochs 50  # Continue for more epochs
"""

# =============================================================================
# MONITORING & METRICS
# =============================================================================

"""
During training, the script logs:
- Training loss & accuracy
- Validation loss & accuracy
- Precision & recall (per class)
- Learning rate
- Best model selection

Metrics are saved to: checkpoints_phase2_full/metrics.json
Format:
{
  "history": {
    "train_loss": [...],
    "train_acc": [...],
    "val_loss": [...],
    "val_acc": [...],
    "train_precision": [...],
    "train_recall": [...]
  },
  "best_val_acc": 0.95,
  "best_val_loss": 0.15
}
"""

# =============================================================================
# NEXT STEPS (PHASE 3 & 4)
# =============================================================================

"""
Phase 3: Geometry-Aware Inpainting
- Take strikeout-detected components (from Phase 2)
- Apply Navier-Stokes interpolation to remove strikeouts
- Output: Cleaned component images

Phase 4: LaTeX Generation  
- Input: Cleaned components from Phase 3
- Use Vision-Language Model (Qwen2.5-VL-32B)
- Output: LaTeX expressions
"""

if __name__ == '__main__':
    print(__doc__)
