"""
Inference script for ViT Strikeout Detector

Provides utilities for loading trained models and making predictions
on new images.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import cv2

from vit_strikeout_detector import (
    ViTStrikeoutDetector,
    StrikeoutDetector,
    GeometryNormalizer
)


class StrikeoutInference:
    """
    Inference pipeline for strikeout detection.
    """
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize inference with trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = ViTStrikeoutDetector(num_classes=2, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.detector = StrikeoutDetector(self.model, device=self.device)
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Device: {self.device}")
    
    def preprocess_image(self, image_path: str, 
                        image_size: int = 224) -> torch.Tensor:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to image
            image_size: Target image size
            
        Returns:
            3-channel input tensor
        """
        # Load image
        if image_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.array(Image.open(image_path).convert('L'))
        else:
            img = np.array(Image.open(image_path).convert('L'))
        
        # Resize
        if img.shape != (image_size, image_size):
            img = cv2.resize(img, (image_size, image_size))
        
        # Normalize
        img = GeometryNormalizer.normalize_image(img)
        
        # Create 3-channel input
        channel_0 = torch.tensor(img, dtype=torch.float32)
        channel_1 = torch.zeros_like(channel_0)  # Default area
        channel_2 = torch.zeros_like(channel_0)  # Default bbox size
        
        x = torch.stack([channel_0, channel_1, channel_2], dim=0).unsqueeze(0)
        
        return x
    
    def predict(self, image_path: str) -> Tuple[int, float]:
        """
        Predict strikeout for image.
        
        Args:
            image_path: Path to image
            
        Returns:
            (prediction, confidence)
            - prediction: 0 (no strikeout) or 1 (strikeout)
            - confidence: Probability of prediction
        """
        x = self.preprocess_image(image_path)
        
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            confidence = probs[0, pred].item()
        
        return pred, confidence
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Predict for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of (prediction, confidence) tuples
        """
        results = []
        for path in image_paths:
            try:
                pred, conf = self.predict(path)
                results.append({
                    'path': path,
                    'prediction': pred,
                    'confidence': conf,
                    'label': 'Strikeout' if pred == 1 else 'No Strikeout'
                })
            except Exception as e:
                results.append({
                    'path': path,
                    'error': str(e)
                })
        
        return results
    
    def get_attention_maps(self, image_path: str) -> torch.Tensor:
        """
        Get attention maps for interpretability.
        
        Args:
            image_path: Path to image
            
        Returns:
            Attention tensor
        """
        x = self.preprocess_image(image_path)
        x = x.to(self.device)
        
        with torch.no_grad():
            attention = self.model.get_attention_maps(x)
        
        return attention


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description="Inference strikeout detector")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                       help='Path to single image')
    parser.add_argument('--image-dir', type=str,
                       help='Directory of images')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize inference
    inf = StrikeoutInference(args.checkpoint, device=args.device)
    
    # Single image
    if args.image:
        print(f"\nProcessing: {args.image}")
        pred, conf = inf.predict(args.image)
        label = 'Strikeout' if pred == 1 else 'No Strikeout'
        print(f"Prediction: {label} (confidence: {conf:.4f})")
    
    # Directory
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        print(f"\nProcessing {len(image_paths)} images...")
        results = inf.predict_batch(image_paths)
        
        for result in results:
            if 'error' not in result:
                print(f"{result['path']}: {result['label']} ({result['confidence']:.4f})")
            else:
                print(f"{result['path']}: ERROR - {result['error']}")


if __name__ == '__main__':
    main()
