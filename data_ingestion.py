"""
data_ingestion.py
=================
Phase 1 — Token Construction Pipeline.

Streams full-page manuscript images from Hugging Face (original_img split),
runs the complete 6-step tokenisation pipeline (tokenization.py), and
saves per-sample results to ./tokenization_results/.

Phase 2 inference (Multimodal Deletion ViT) will be re-integrated here
once training is complete and best_model.pth is available.
"""

import os
from dotenv import load_dotenv
from datasets import load_dataset
from tokenization import process_pil_image
from pathlib import Path
from tqdm import tqdm

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_token")


def ingest_and_tokenize(
    dataset_name = "Incinciblecolonel/MathStrike",
    data_dir     = "original_img",
    limit        = 10,
):
    """
    Streams `limit` images from the HuggingFace dataset,
    applies the 6-step Token Construction pipeline,
    and saves results to ./tokenization_results/.

    Output per sample (in tokenization_results/sample_XXXXXX/):
        tokens.json        — component metadata + token groups
        pipeline_steps.png — visualisation of all 6 steps
        crops/             — individual token crop images
    """
    print(f"Loading dataset: {dataset_name} (dir: {data_dir})...")

    try:
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

        for i, sample in enumerate(tqdm(dataset, total=limit)):
            if i >= limit:
                break

            # Retrieve image (column name is 'image' for this dataset)
            image = sample.get("image")
            if image is None:
                for key, value in sample.items():
                    if hasattr(value, "size"):   # PIL image duck-typing
                        image = value
                        break

            if image is None:
                print(f"Warning: No image found in sample {i}")
                continue

            # ── Phase 1: Token Construction ───────────────────────────────
            process_pil_image(
                pil_image  = image,
                sample_id  = i,
                output_dir = output_dir,
                save_vis   = True,
                save_crops = True,
                verbose    = False,
            )

        print(f"\nPhase 1 complete! Results saved to: {output_dir.absolute()}")
        print("Next step: run train_vit.py to train the Deletion ViT.")

    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise


if __name__ == "__main__":
    ingest_and_tokenize(limit=5)
