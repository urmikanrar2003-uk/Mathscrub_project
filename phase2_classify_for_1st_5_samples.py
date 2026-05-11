"""
phase2_classify.py
==================
Phase 2 — Multimodal Deletion Classification.

Reads tokens.json from Phase 1, builds the 3-channel tensor for each token,
runs it through the trained MultimodalDeletionViT (best_model.pth),
and saves classification_results.json per sample.

HOW TO RUN:
    python phase2_classify.py

REQUIRES:
    - ./tokenization_results/sample_*/tokens.json   (from Phase 1)
    - ./tokenization_results/sample_*/crops/        (crop images from Phase 1)
    - ./best_model.pth                              (from train_vit.py)
"""

import json
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from vit_strikeout_detector import MultimodalDeletionViT, pad_to_square, MU_A, SIGMA_A, MU_S, SIGMA_S

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH       = "./best_model.pth"
RESULTS_DIR      = Path("./tokenization_results")
DELETE_THRESHOLD = 0.5   # p_delete >= 0.5 → "remove", else "retain"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load the trained model
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_path: str) -> MultimodalDeletionViT:
    print(f"Loading model from: {model_path}")
    model = MultimodalDeletionViT(pretrained=False)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on: {DEVICE}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Build 3-channel tensor from one token group
# ═══════════════════════════════════════════════════════════════════════════

def build_token_tensor(crop_path: Path, box_area: float, comp_size: float) -> torch.Tensor:
    """
    Builds the (3, 224, 224) early-fusion tensor for one token.

    Channel 0 — Grayscale crop, padded to square, normalized to [0,1]
    Channel 1 — X_area = (log(box_area+1) - MU_A) / SIGMA_A  broadcast
    Channel 2 — X_size = (log(comp_size+1) - MU_S) / SIGMA_S broadcast
    """
    img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read crop: {crop_path}")

    img_padded = pad_to_square(img, target_size=224, padding_value=255)
    ch0 = img_padded.astype(np.float32) / 255.0

    x_area = (np.log(box_area + 1) - MU_A) / SIGMA_A
    ch1 = np.full((224, 224), x_area, dtype=np.float32)

    x_size = (np.log(comp_size + 1) - MU_S) / SIGMA_S
    ch2 = np.full((224, 224), x_size, dtype=np.float32)

    return torch.from_numpy(np.stack([ch0, ch1, ch2], axis=0))


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3 — Classify all tokens in one sample folder
# ═══════════════════════════════════════════════════════════════════════════

def classify_sample(sample_dir: Path, model: MultimodalDeletionViT) -> dict:
    """
    tokens.json structure (from your Phase 1):
        {
          "tokens": [[0], [1], [5, 6], ...],   <- groups of component IDs
          "components": [
            {"id": 0, "area": 8561, "bbox": [x, y, w, h], ...},
            ...
          ]
        }

    For each token group, we:
      1. Take the FIRST component ID in the group
      2. Look up its area and bbox from components
      3. Find its crop image in crops/
      4. Build the 3-channel tensor and run inference
    """
    tokens_path = sample_dir / "tokens.json"
    crops_dir   = sample_dir / "crops"

    if not tokens_path.exists():
        print(f"  WARNING: No tokens.json in {sample_dir}, skipping.")
        return {}

    with open(tokens_path, "r") as f:
        data = json.load(f)

    # Build a lookup: component_id -> component metadata
    components_list = data.get("components", [])
    comp_lookup = {c["id"]: c for c in components_list}

    # tokens is a list of lists e.g. [[0], [1], [5, 6], ...]
    token_groups = data.get("tokens", [])

    tensors = []
    meta    = []
    skipped = []

    for token_idx, group in enumerate(token_groups):
        # Use the first component in the group as the representative
        primary_id = group[0]
        comp = comp_lookup.get(primary_id)

        if comp is None:
            print(f"  WARNING: Component {primary_id} not found in sample {sample_dir.name}")
            skipped.append({"token_idx": token_idx, "component_ids": group, "label": "unknown"})
            continue

        # area is directly in the component
        box_area = float(comp.get("area", 1.0))

        # bbox format is [x, y, w, h] -- comp_size = max(w, h)
        bbox = comp.get("bbox", [0, 0, 1, 1])
        w, h = bbox[2], bbox[3]
        comp_size = float(max(w, h))

        # Find the crop image -- named by token index (from tokenization.py)
        crop_candidates = [
            crops_dir / f"token_{token_idx:04d}.png",
            crops_dir / f"token_{token_idx}.png",
            # Fallbacks just in case
            crops_dir / f"component_{primary_id:04d}.png",
            crops_dir / f"component_{primary_id}.png",
            crops_dir / f"{primary_id:04d}.png",
            crops_dir / f"{primary_id}.png",
        ]
        crop_path = next((p for p in crop_candidates if p.exists()), None)

        if crop_path is None:
            # Print one example of what crops are named so we can fix if needed
            if token_idx == 0 and crops_dir.exists():
                all_crops = list(crops_dir.iterdir())
                if all_crops:
                    print(f"  INFO: Example crop filename: {all_crops[0].name}")
            skipped.append({
                "token_idx":     token_idx,
                "component_ids": group,
                "label":         "no_crop_found"
            })
            continue

        try:
            t = build_token_tensor(crop_path, box_area, comp_size)
            tensors.append(t)
            meta.append({
                "token_idx":     token_idx,
                "component_ids": group,
                "primary_id":    primary_id,
                "crop_path":     str(crop_path),
                "box_area":      box_area,
                "comp_size":     comp_size,
            })
        except Exception as e:
            print(f"  WARNING: Failed tensor for component {primary_id}: {e}")
            skipped.append({
                "token_idx":     token_idx,
                "component_ids": group,
                "label":         "error"
            })

    # --- Batch inference ---
    classified = []
    if tensors:
        batch = torch.stack(tensors).to(DEVICE)   # (N, 3, 224, 224)
        with torch.no_grad():
            p_deletes = model(batch).cpu().numpy()  # (N,)

        for j, p in enumerate(p_deletes):
            classified.append({
                "token_idx":     meta[j]["token_idx"],
                "component_ids": meta[j]["component_ids"],
                "primary_id":    meta[j]["primary_id"],
                "p_delete":      round(float(p), 4),
                "label":         "remove" if p >= DELETE_THRESHOLD else "retain",
                "crop_path":     meta[j]["crop_path"],
            })

    all_results = sorted(classified + skipped, key=lambda x: x["token_idx"])

    return {
        "sample":       sample_dir.name,
        "total_tokens": len(token_groups),
        "classified":   len(classified),
        "skipped":      len(skipped),
        "tokens":       all_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4 — Run Phase 2 across all samples
# ═══════════════════════════════════════════════════════════════════════════

def run_phase2(results_dir: Path = RESULTS_DIR, model_path: str = MODEL_PATH):
    model = load_model(model_path)

    sample_dirs = sorted(results_dir.glob("sample_*"))[:5]
    if not sample_dirs:
        print(f"No sample folders found in {results_dir}. Run Phase 1 first.")
        return

    print(f"\nRunning Phase 2 on {len(sample_dirs)} samples...\n")

    all_results = []
    for sample_dir in tqdm(sample_dirs):
        result = classify_sample(sample_dir, model)
        if not result:
            continue

        out_path = sample_dir / "classification_results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        tokens    = result.get("tokens", [])
        n_remove  = sum(1 for t in tokens if t.get("label") == "remove")
        n_retain  = sum(1 for t in tokens if t.get("label") == "retain")
        n_skipped = result.get("skipped", 0)
        print(f"  {sample_dir.name}: {len(tokens)} tokens -> {n_remove} remove | {n_retain} retain | {n_skipped} skipped")

        all_results.append(result)

    summary_path = results_dir / "phase2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nPhase 2 complete!")
    print(f"  Per-sample : <sample_dir>/classification_results.json")
    print(f"  Summary    : {summary_path}")
    print(f"\nNext step: run geometry_inpainting.py")


if __name__ == "__main__":
    run_phase2()
