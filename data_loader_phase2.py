"""
Data loading utilities for strikeout detection dataset.

Handles loading and preprocessing real dataset with component annotations.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrikeoutComponentDataset(Dataset):
    """
    Dataset for strikeout detection using component-level annotations.
    
    Loads components from full page images using bounding box annotations.
    """
    
    def __init__(self, annotations: List[Dict], 
                 page_img_dir: str, 
                 image_size: int = 224,
                 normalize_area: bool = True):
        """
        Initialize dataset.
        
        Args:
            annotations: List of annotation dictionaries from JSON
            page_img_dir: Directory containing full page images
            image_size: Target image size
            normalize_area: Whether to normalize geometry features
        """
        self.annotations = annotations
        self.page_img_dir = Path(page_img_dir)
        self.image_size = image_size
        self.normalize_area = normalize_area
        
        # Cache for loaded page images
        self.page_cache = {}
        
        # Collect unique labels for statistics
        self.labels_count = {}
        for ann in annotations:
            label = ann.get('label', 0)
            self.labels_count[label] = self.labels_count.get(label, 0) + 1
        
        logger.info(f"Dataset initialized with {len(annotations)} samples")
        logger.info(f"Label distribution: {self.labels_count}")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def _load_page_image(self, page_img: str) -> Optional[np.ndarray]:
        """Load page image with caching."""
        if page_img in self.page_cache:
            return self.page_cache[page_img]
        
        img_path = self.page_img_dir / page_img
        
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            return None
        
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.page_cache[page_img] = img
            return img
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            return None
    
    def _extract_component(self, image: np.ndarray, 
                          bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract component from image using bounding box.
        
        Args:
            image: Full page image
            bbox: [x, y, height, width] bounding box
            
        Returns:
            Cropped component image or None if extraction fails
        """
        if image is None:
            return None
        
        try:
            x, y, h, w = bbox
            
            # Clamp coordinates to image bounds
            x = max(0, x)
            y = max(0, y)
            x_end = min(image.shape[1], x + w)
            y_end = min(image.shape[0], y + h)
            
            # Ensure valid crop
            if x_end <= x or y_end <= y:
                return None
            
            component = image[y:y_end, x:x_end]
            
            # Resize to target size
            if component.shape != (self.image_size, self.image_size):
                component = cv2.resize(component, 
                                      (self.image_size, self.image_size))
            
            return component
        except Exception as e:
            logger.warning(f"Error extracting component: {e}")
            return None
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1]."""
        if image is None:
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        img_min = image.min()
        img_max = image.max()
        
        if img_max == img_min:
            return np.zeros_like(image, dtype=np.float32)
        
        return ((image - img_min) / (img_max - img_min)).astype(np.float32)
    
    def _normalize_geometry(self, ann: Dict, page_img: Optional[np.ndarray]) -> Tuple[float, float]:
        """Compute normalized geometry features."""
        box_area = ann.get('box_area', 1comp_size[1]
        norm_size = np.sqrt((comp_h**2 + comp_w**2) / 2) / max(self.image_size, 224)
        norm_size = min(1.0, norm_size)
        
        return norm_area, norm_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (3-channel input tensor, label)
        """)
        comp_size = ann.get('comp_size', [1, 1])
        
        # Get page image dimensions if available
        if page_img is not None:
            page_area = page_img.shape[0] * page_img.shape[1]
        else:
            page_area = 1000000  # Default assumption
        
        # Normalize box area
        norm_area = min(1.0, box_area / page_area) if page_area > 0 else 0.0
        
        # Normalize component size
        comp_h, comp_w = comp_size[0], comp_size[1]
        norm_size = np.sqrt((comp_h**2 + comp_w**2) / 2) / max(self.image_size, 224)
        norm_size = min(1.0, norm_size)
        
        return norm_area, norm_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (3-channel input tensor, label)
        """
        ann = self.annotations[idx]
        
        # Load page image
        page_img = self._load_page_image(ann['page_img'])
        
        # Extract component
        component = self._extract_component(page_img, ann['bbox'])
        
        if component is None:
            # Return dummy data if extraction fails
            component = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Normalize image
        component = self._normalize_image(component)
        
        # Get geometry features
        norm_area, norm_size = self._normalize_geometry(ann, page_img)
        
        # Create 3-channel input
        channel_0 = torch.tensor(component, dtype=torch.float32)
        channel_1 = torch.full_like(channel_0, norm_area)
        channel_2 = torch.full_like(channel_0, norm_size)
        
        x = torch.stack([channel_0, channel_1, channel_2], dim=0)
        
        # Get label
        label = torch.tensor(ann.get('label', 0), dtype=torch.long)
        
        return x, label


class DatasetBuilder:
    """Utility for building datasets from annotation files."""
    
    @staticmethod
    def from_json(json_path: str, 
                  page_img_dir: str,
                  image_size: int = 224) -> StrikeoutComponentDataset:
        """
        Build dataset from JSON annotation file.
        
        Args:
            json_path: Path to annotation JSON file
            page_img_dir: Directory containing page images
            image_size: Target image size
            
        Returns:
            StrikeoutComponentDataset instance
        """
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        logger.info(f"Loaded {len(annotations)} annotations from {json_path}")
        
        return StrikeoutComponentDataset(
            annotations=annotations,
            page_img_dir=page_img_dir,
            image_size=image_size
        )
    
    @staticmethod
    def create_dataloaders(
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        use_synth: bool = False
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and test dataloaders.
        
        Args:
            data_dir: Base data directory (e.g., /path/to/archive)
            batch_size: Batch size
            num_workers: Number of dataloader workers
            image_size: Target image size
            use_synth: Whether to use synthetic data
            
        Returns:
            (train_loader, test_loader) tuple
        """
        data_path = Path(data_dir)
        
        # Determine which dataset to use
        if use_synth:
            dataset_dir = data_path / 'synth'
        else:
            dataset_dir = data_path / 'real'
        
        logger.info(f"Using dataset: {dataset_dir}")
        
        # Check required files
        train_json = dataset_dir / 'train_split.json'
        test_json = dataset_dir / 'test_split.json'
        page_img_dir = data_path / 'original_img'
        
        if not train_json.exists():
            raise FileNotFoundError(f"Train split not found: {train_json}")
        if not test_json.exists():
            raise FileNotFoundError(f"Test split not found: {test_json}")
        if not page_img_dir.exists():
            raise FileNotFoundError(f"Page image directory not found: {page_img_dir}")
        
        # Load datasets
        train_dataset = DatasetBuilder.from_json(
            str(train_json),
            str(page_img_dir),
            image_size=image_size
        )
        
        test_dataset = DatasetBuilder.from_json(
            str(test_json),
            str(page_img_dir),
            image_size=image_size
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created train loader: {len(train_loader)} batches")
        logger.info(f"Created test loader: {len(test_loader)} batches")
        
        return train_loader, test_loader


if __name__ == '__main__':
    """Test data loading"""
    print("Testing data loading...")
    
    # Create dataloaders
    train_loader, test_loader = DatasetBuilder.create_dataloaders(
        data_dir='./data',
        batch_size=32,
        num_workers=0,
        image_size=224,
        use_synth=False
    )
    
    # Test first batch
    print("\nTesting first batch from train loader...")
    for x, y in train_loader:
        print(f"Batch shape: {x.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Label distribution: {torch.bincount(y)}")
        break
    
    # Test test loader
    print("\nTesting first batch from test loader...")
    for x, y in test_loader:
        print(f"Batch shape: {x.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Label distribution: {torch.bincount(y)}")
        break
    
    print("\n✓ Data loading test passed!")
