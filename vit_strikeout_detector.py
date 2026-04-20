import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import json
import glob

def pad_to_square(img_crop, target_size=224, padding_value=255):
    """
    Pads a grayscale crop to make it square, preserving aspect ratio,
    then resizes to the target_size.
    """
    h, w = img_crop.shape[:2]
    max_dim = max(h, w)
    
    # Calculate padding
    pad_h = max_dim - h
    pad_w = max_dim - w
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # Pad image
    squared_img = cv2.copyMakeBorder(
        img_crop, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=padding_value
    )
    
    # Resize to target size
    resized_img = cv2.resize(squared_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized_img


def prepare_multimodal_tensor(
    component, 
    gray_image, 
    mu_a=8.7570, sigma_a=0.7901, 
    mu_s=5.1571, sigma_s=0.5591, 
    target_size=224
):
    """
    Prepares the 3-channel input representation for a single component.
    Channel 0: Grayscale image crop, squared and resized.
    Channel 1: Spatially broadcast X_area scalar.
    Channel 2: Spatially broadcast X_size scalar.
    """
    x, y, w, h = component["bbox"]
    area = component.get("area", w * h)
    
    # Safety boundary checks
    img_h, img_w = gray_image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    
    # Crop the component
    crop = gray_image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.full((10, 10), 255, dtype=np.uint8)
        
    # Standardize image
    resized_crop = pad_to_square(crop, target_size=target_size, padding_value=255)
    img_channel = resized_crop.astype(np.float32) / 255.0  # Normalize to [0,1]
    
    # Compute geometry scalars
    # 1. Area
    log_area = math.log(area + 1)
    x_area = (log_area - mu_a) / sigma_a
    
    # 2. Size (Max dimension of bounding box)
    L = max(w, h)
    log_size = math.log(L + 1)
    x_size = (log_size - mu_s) / sigma_s
    
    # Create broadcasted channels
    area_channel = np.full((target_size, target_size), x_area, dtype=np.float32)
    size_channel = np.full((target_size, target_size), x_size, dtype=np.float32)
    
    # Stack into (3, H, W) tensor
    multimodal_tensor = np.stack([img_channel, area_channel, size_channel], axis=0)
    return torch.from_numpy(multimodal_tensor)


class MultimodalDeletionViT(nn.Module):
    def __init__(self, pretrained=True):
        super(MultimodalDeletionViT, self).__init__()
        # Load a base ViT
        # We can use pretrained weights, although channel 1 & 2 are no longer RGB.
        # It serves as a decent starting point.
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = vit_b_16(weights=weights)
        
        # Replace the head with a binary classification head
        hidden_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        # Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (Batch, 3, 224, 224)
        Returns probabilities of shape (Batch,)
        """
        logits = self.vit(x)
        probs = self.sigmoid(logits)
        return probs.squeeze(-1)


def predict_deletion_components(
    components, 
    gray_image, 
    model, 
    device="cpu", 
    mu_a=8.7570, sigma_a=0.7901, 
    mu_s=5.1571, sigma_s=0.5591, 
    threshold=0.5
):
    """
    Given a list of component dictionaries and the full grayscale image,
    runs the Multimodal ViT model for each connected component.
    """
    model.eval()
    model.to(device)
    
    results = []
    
    with torch.no_grad():
        for comp in components:
            # 1. Prepare tensor
            tensor_3ch = prepare_multimodal_tensor(
                comp, gray_image,
                mu_a=mu_a, sigma_a=sigma_a,
                mu_s=mu_s, sigma_s=sigma_s,
                target_size=224
            )
            # Add batch dimension: (1, 3, 224, 224)
            batch_input = tensor_3ch.unsqueeze(0).to(device)
            
            # 2. Forward pass
            prob = model(batch_input).item()
            
            # 3. Decision
            decision = "remove" if prob >= threshold else "retain"
            
            results.append({
                "component_id": comp["id"],
                "p_delete": prob,
                "decision": decision
            })
            
    return results

if __name__ == "__main__":
    # Small test script to verify logic and dimensions
    print("Testing MultimodalDeletionViT with synthetic data...")
    
    # Dummy components
    components = [
        {"id": 0, "bbox": (10, 10, 50, 50), "area": 1200}, # A math symbol
        {"id": 1, "bbox": (20, 20, 300, 400), "area": 80000} # A huge deletion stroke
    ]
    
    # Dummy grayscale image
    dummy_img = np.full((800, 800), 255, dtype=np.uint8)
    
    # Initialize model
    model = MultimodalDeletionViT(pretrained=False)
    
    # Run predictions
    predictions = predict_deletion_components(
        components, 
        dummy_img, 
        model, 
        device="cpu",
        mu_a=8.7570, sigma_a=0.7901,
        mu_s=5.1571, sigma_s=0.5591
    )
    
    for p in predictions:
        print(f"Component {p['component_id']} -> p_delete: {p['p_delete']:.4f} ({p['decision']})")
    
    print("Test passed successfully. Data tensor shapes and model operations are correct.")
