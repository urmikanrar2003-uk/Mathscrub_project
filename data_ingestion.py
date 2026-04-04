import os
from dotenv import load_dotenv
from datasets import load_dataset
from tokenization import process_pil_image
from pathlib import Path
from tqdm import tqdm

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
                
            process_pil_image(
                pil_image=image,
                sample_id=i,
                output_dir=output_dir,
                save_vis=True,
                save_crops=True,
                verbose=False
            )
            
        print(f"\nProcessing complete! Results saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    # You can increase the limit as needed
    ingest_and_tokenize(limit=5)
