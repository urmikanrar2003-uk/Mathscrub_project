import os
from dotenv import load_dotenv
from datasets import load_dataset
from tokenization import process_pil_image
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import json
import torch
from vit_strikeout_detector import MultimodalDeletionViT, predict_deletion_components

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_token")

def ingest_and_tokenize(dataset_name="Incinciblecolonel/MathStrike", data_dir="original_img", limit=5):
    """
    Ingests data from Hugging Face and applies tokenization.
    """
    print(f"Loading dataset: {dataset_name} (dir: {data_dir})...")
    
    try:
        # Load the dataset
        # We use token for private or rate-limited access if needed
        dataset = load_dataset(
            dataset_name, 
            data_dir=data_dir, 
            split="train", 
            streaming=True, 
            token=hf_token
        )
        
        print("Initializing Vision Transformer...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MultimodalDeletionViT(pretrained=False)
        model.to(device)
        
        output_dir = Path("./tokenization_results")
        output_dir.mkdir(exist_ok=True)
        
        print(f"Processing first {limit} samples...")
        
        # Note: streaming=True doesn't provide total length unless known
        for i, sample in enumerate(tqdm(dataset, total=limit)):
            if i >= limit:
                break
                
            # Access the image. The column name is usually 'image' for image datasets
            image = sample.get("image")
            if image is None:
                # Fallback: check all keys for a PIL image
                for key, value in sample.items():
                    if hasattr(value, "size"): # Basic PIL image attribute
                        image = value
                        break
            
            if image is None:
                print(f"Warning: No image found in sample {i}")
                continue
                
            # 1. Run Tokenization Pipeline
            meta_data = process_pil_image(
                pil_image=image,
                sample_id=i,
                output_dir=output_dir,
                save_vis=True,
                save_crops=True,
                verbose=False
            )
            
            # 2. Extract components & Convert to Grayscale
            components = meta_data.get("components", [])
            gray_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
            
            # 3. Predict Deletions via ViT
            predictions = predict_deletion_components(
                components=components, 
                gray_image=gray_image, 
                model=model, 
                device=device
            )
            
            # 4. Save predictions alongside tokenization metadata
            meta_data["vit_predictions"] = predictions
            
            sample_dir = output_dir / f"sample_{i:06d}"
            with open(sample_dir / "tokens.json", "w") as f:
                json.dump(meta_data, f, indent=2)
            
        print(f"\nProcessing complete! Results saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    # You can increase the limit as needed
    ingest_and_tokenize(limit=5)
