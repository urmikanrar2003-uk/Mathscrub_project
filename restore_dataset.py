"""
restore_dataset.py
==================
Unified Full-Dataset Restoration Pipeline for MathScrub.

Streams ALL images from Incinciblecolonel/MathStrike (original_img split),
runs Phases 1-3 entirely in memory per sample, and saves ONLY:
    - restored_img_final.png   (cleaned image for Phase 4 fine-tuning)
    - meta.json                (slim metadata: sample_id, image_shape, num_tokens)

Temporary files (crops/, pipeline_steps.png) are NEVER written to disk.

OUTPUT DIRECTORY: D:\\MathScrub_Restored\\

RESUMABLE: If restored_img_final.png already exists for a sample, it is skipped.

HOW TO RUN:
    python restore_dataset.py

    # To limit samples for testing (e.g. first 20):
    python restore_dataset.py --limit 20

    # To use a custom output directory:
    python restore_dataset.py --output_dir "E:\\MyOutput"
"""

import os
import sys
import json
import shutil
import argparse
import traceback
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset

# ── Local imports ─────────────────────────────────────────────────────────────
from tokenization import token_construction
from vit_strikeout_detector import (
    MultimodalDeletionViT, pad_to_square,
    MU_A, SIGMA_A, MU_S, SIGMA_S
)
from geometry_inpainting import inpaint_math_equation


# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN         = os.getenv("HF_token")
DATASET_NAME     = "Incinciblecolonel/MathStrike"
DATA_DIR         = "original_img"
MODEL_PATH       = "./best_model.pth"
DEFAULT_OUT_DIR  = Path("D:/MathScrub_Restored")
DELETE_THRESHOLD = 0.5
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str) -> MultimodalDeletionViT:
    print(f"Loading ViT model from: {model_path}")
    model = MultimodalDeletionViT(pretrained=False)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on: {DEVICE}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  BUILD TENSOR FROM NUMPY CROP (no file I/O needed)
# ═════════════════════════════════════════════════════════════════════════════

def build_tensor_from_array(crop_bgr: np.ndarray, box_area: float, comp_size: float) -> torch.Tensor:
    """
    Builds the (3, 224, 224) early-fusion tensor directly from a numpy crop.
    Avoids writing crops to disk.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    padded = pad_to_square(gray, target_size=224, padding_value=255)
    ch0 = padded.astype(np.float32) / 255.0

    x_area = (np.log(box_area + 1) - MU_A) / SIGMA_A
    ch1 = np.full((224, 224), x_area, dtype=np.float32)

    x_size = (np.log(comp_size + 1) - MU_S) / SIGMA_S
    ch2 = np.full((224, 224), x_size, dtype=np.float32)

    return torch.from_numpy(np.stack([ch0, ch1, ch2], axis=0))


# ═════════════════════════════════════════════════════════════════════════════
#  PER-SAMPLE PIPELINE (entirely in RAM)
# ═════════════════════════════════════════════════════════════════════════════

def process_sample(
    pil_image: Image.Image,
    sample_id: int,
    model: MultimodalDeletionViT,
    output_dir: Path,
) -> bool:
    """
    Runs Phases 1-3 entirely in memory for a single image.
    Saves only restored_img_final.png and meta.json.
    Returns True on success, False on failure.
    """

    # ── Convert PIL to BGR ────────────────────────────────────────────────
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    H, W = image_bgr.shape[:2]

    # ── Phase 1: Tokenize in RAM (no save_crops, no save_vis) ────────────
    (tokens, components, all_components,
     retained_edges, rejected_edges,
     binary, label_map, debug) = token_construction(image_bgr, verbose=False)

    if not tokens:
        return False  # Nothing to process

    # ── Phase 2: Build tensors in RAM and classify ────────────────────────
    comp_lookup = {c["id"]: c for c in components}
    tensors = []
    meta    = []

    for t_idx, group in enumerate(tokens):
        primary_id = group[0]
        comp = comp_lookup.get(primary_id)
        if comp is None:
            continue

        box_area  = float(comp.get("area", 1.0))
        bbox      = comp.get("bbox", [0, 0, 1, 1])
        comp_size = float(max(bbox[2], bbox[3]))

        # Crop directly from the original image (no file save)
        bboxes = [comp_lookup[cid]["bbox"] for cid in group if cid in comp_lookup]
        if not bboxes:
            continue
        x1 = max(0, min(b[0] for b in bboxes) - 5)
        y1 = max(0, min(b[1] for b in bboxes) - 5)
        x2 = min(W, max(b[0] + b[2] for b in bboxes) + 5)
        y2 = min(H, max(b[1] + b[3] for b in bboxes) + 5)
        crop_bgr = image_bgr[y1:y2, x1:x2]

        if crop_bgr.size == 0:
            continue

        try:
            t = build_tensor_from_array(crop_bgr, box_area, comp_size)
            tensors.append(t)
            meta.append({"token_idx": t_idx, "component_ids": group, "primary_id": primary_id})
        except Exception:
            continue
        finally:
            # Explicitly free crop memory
            del crop_bgr

    # ── Batch inference ───────────────────────────────────────────────────
    token_predictions = {}   # token_idx -> p_delete
    if tensors:
        batch = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            p_deletes = model(batch).cpu().numpy()
        
        # --- VRAM CLEANUP ---
        del batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for j, p in enumerate(p_deletes):
            token_predictions[meta[j]["token_idx"]] = float(p)

    # ── Build component-level predictions for inpainting ─────────────────
    comp_preds = []
    for comp in components:
        found_token_idx = -1
        for t_idx, group in enumerate(tokens):
            if comp["id"] in group:
                found_token_idx = t_idx
                break

        p = token_predictions.get(found_token_idx, 0.0)
        comp_preds.append({
            "component_id": comp["id"],
            "p_delete": p,
        })

    # ── Phase 3: Inpaint in RAM ───────────────────────────────────────────
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    restored_gray = inpaint_math_equation(gray, components, comp_preds, tau_p=DELETE_THRESHOLD)

    # ── Save outputs ──────────────────────────────────────────────────────
    sample_dir = output_dir / f"sample_{sample_id:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 1. Restored image
    cv2.imwrite(str(sample_dir / "restored_img_final.png"), restored_gray)

    # 2. Slim metadata JSON (only what Phase 4 needs)
    n_remove = sum(1 for p in comp_preds if p["p_delete"] >= DELETE_THRESHOLD)
    slim_meta = {
        "sample_id":    sample_id,
        "image_shape":  [H, W],
        "num_tokens":   len(tokens),
        "num_removed":  n_remove,
        "num_retained": len(tokens) - n_remove,
    }
    with open(sample_dir / "meta.json", "w") as f:
        json.dump(slim_meta, f, indent=2)

    return True


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(output_dir: Path, limit: int = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_log = output_dir / "failed_samples.txt"
    progress_log = output_dir / "progress.txt"

    # Load already-processed sample IDs for resumption
    processed_ids = set()
    if progress_log.exists():
        with open(progress_log, "r") as f:
            processed_ids = set(int(line.strip()) for line in f if line.strip().isdigit())
    print(f"Resuming: {len(processed_ids)} samples already processed.")

    # Load model once
    model = load_model(MODEL_PATH)

    # Stream dataset from Hugging Face
    print(f"Streaming dataset: {DATASET_NAME} / {DATA_DIR}")
    dataset = load_dataset(
        DATASET_NAME,
        data_dir=DATA_DIR,
        split="train",
        streaming=True,
        token=HF_TOKEN,
    )

    total_done    = 0
    total_skipped = 0
    total_failed  = 0

    progress_f = open(progress_log, "a")
    failed_f   = open(failed_log, "a")

    try:
        for i, sample in enumerate(tqdm(dataset, desc="Processing dataset")):
            if limit is not None and i >= limit:
                break

            # Skip if already done (resumption support)
            if i in processed_ids:
                total_skipped += 1
                continue

            # Get image
            image = sample.get("image")
            if image is None:
                for v in sample.values():
                    if hasattr(v, "size"):
                        image = v
                        break

            if image is None:
                failed_f.write(f"{i}\tno_image\n")
                failed_f.flush()
                total_failed += 1
                continue

            try:
                success = process_sample(image, i, model, output_dir)
                if success:
                    progress_f.write(f"{i}\n")
                    progress_f.flush()
                    total_done += 1
                else:
                    failed_f.write(f"{i}\tempty_tokens\n")
                    failed_f.flush()
                    total_failed += 1
            except Exception as e:
                err_msg = traceback.format_exc()
                failed_f.write(f"{i}\t{str(e)[:100]}\n")
                failed_f.flush()
                total_failed += 1
                tqdm.write(f"  [ERROR] sample_{i:06d}: {e}")
                continue

            if (i + 1) % 100 == 0:
                tqdm.write(
                    f"  Checkpoint: done={total_done} | skipped={total_skipped} | failed={total_failed}"
                )

    finally:
        progress_f.close()
        failed_f.close()

    print(f"\nPipeline complete!")
    print(f"  Done    : {total_done}")
    print(f"  Skipped : {total_skipped}")
    print(f"  Failed  : {total_failed}")
    print(f"  Output  : {output_dir.absolute()}")
    if total_failed > 0:
        print(f"  Failed log: {failed_log}")


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MathScrub Full Dataset Restoration Pipeline")
    parser.add_argument(
        "--output_dir", type=str, default=str(DEFAULT_OUT_DIR),
        help="Directory on D: drive to save results (default: D:/MathScrub_Restored)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples (for testing). Default: process all."
    )
    args = parser.parse_args()

    run_pipeline(
        output_dir=Path(args.output_dir),
        limit=args.limit,
    )
