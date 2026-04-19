# Mathscrub Project

Single-pipeline strikeout removal and LaTeX generation for handwritten calculus expressions.

## 📌 Status

🚧 **Development Phase** — Features and APIs may change

### Project Phases

| Phase | Module | Description | Status |
|-------|--------|-------------|--------|
| **Phase 1** | Token Construction | Delaunay triangulation and Union-Find grouping for semantic symbol clustering | ✅ **COMPLETE** |
| **Phase 2** | Multimodal Deletion Classification | Vision Transformer (ViT) for strikeout detection and classification | 🔄 In Progress |
| **Phase 3** | Geometry-Aware Inpainting | Image restoration using Navier-Stokes interpolation after strikeout removal | 📋 Planned |
| **Phase 4** | LaTeX Generation | Vision-Language Model (Qwen2.5-VL-32B) transcription with Parameter-Efficient Fine-Tuning | 📋 Planned |

## 📖 Overview

Mathscrub is a recreation of the research paper **"MathScrub: Single-Pipeline Strikeout Removal and LaTeX Generation for Handwritten Calculus Expressions"** (Sen Shen et al., IEEE Transactions on Learning Technologies, 2026).

This project is designed to extract and tokenize mathematical symbols and structures from handwritten document images. It ingests data from Hugging Face datasets and applies a sophisticated four-module pipeline to process handwritten calculus expressions, ultimately generating LaTeX code.

### Key Features

- **Data Ingestion**: Automatically loads image datasets from Hugging Face
- **Image Tokenization**: Converts mathematical images into semantic tokens using Delaunay triangulation and Union-Find clustering *(Phase 1 Complete)*
- **Structured Output**: Generates tokens and cropped regions for further analysis
- **[In Development] Strikeout Detection**: Uses Vision Transformers for deletion classification
- **[In Development] Image Restoration**: Geometry-aware inpainting after strikeout removal
- **[In Development] LaTeX Transcription**: Vision-Language Model transcription with PEFT

## 🔧 MathScrub Framework (4 Modules)

### Phase 1: Token Construction ✅ (COMPLETE)

The tokenization process follows exact algorithms from the MathScrub paper with six main steps:

1. **Binarization** — Convert images to binary (black ink on white background) using Otsu's adaptive thresholding
2. **Connected Components** — Identify separate components in the binary image
3. **Delaunay Triangulation** — Build geometric relationships between components using exact Euclidean distance and inclination constraints
4. **Nested Suppression** — Remove redundant or nested components
5. **Grouping (Union-Find)** — Group related components into semantic units using transitive merging
6. **Edge Filtering** — Refine and filter edges using horizontal overlap ratio and area ratio constraints

**Output**: Semantically coherent groups of components (tokens)

### Phase 2: Multimodal Deletion Classification 🚀 (IN PROGRESS)

**Architecture: Vision Transformer (ViT-Base)**
- Model: google/vit-base-patch16-224 (86.7M parameters)
- Input: 3-channel multimodal data
  - Channel 1: Grayscale component image (224×224)
  - Channel 2: Component area normalized by page area
  - Channel 3: Component size normalized by diagonal
- Output: Binary classification (strikeout / no-strikeout)
- Task: Detect strikeouts in handwritten mathematical components

**Dataset Integration**
- Real data: 97,223 training components + 1,000 balanced test samples
- Automated component extraction from full page images using bounding boxes
- Geometry-aware feature normalization
- On-the-fly image caching for training efficiency

**Training Pipeline**
- Optimizer: AdamW with weight decay (1e-5) and linear warmup
- Learning rate: 1e-4 with cosine annealing scheduling
- Batch size: 64 (configurable based on GPU)
- Validation metrics: Accuracy, Precision, Recall
- Checkpoint saving (best model + periodic)

**Files & Scripts**
- `vit_strikeout_detector.py` — ViT model architecture & inference API
- `data_loader_phase2.py` — Dataset loading with component extraction
- `train_phase2_real_data.py` — Full training pipeline
- `train_demo.py` — Quick demo training (5K samples, 5 epochs)
- `inference_strikeout_detector.py` — Inference on new images
- `PHASE2_GUIDE.py` — Comprehensive guide & commands

### Phase 3: Geometry-Aware Inpainting (Planned)

- Applies Navier-Stokes interpolation for boundary-aware restoration
- Uses feathering techniques for seamless inpainting of strikeout regions
- Precise size thresholds to separate small gaps and larger regions
- Restores image quality after strikeout removal

### Phase 4: LaTeX Generation (Planned)

- Vision-Language Model: Qwen2.5-VL-32B for expression transcription
- Parameter-Efficient Fine-Tuning (PEFT) strategy with adaptive LoRA
- Adaptive LoRA on vision stack, fixed-rank LoRA on text decoder
- Generates accurate LaTeX code for mathematical expressions

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Mathscrub_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
Create a `.env` file in the project root with your Hugging Face token:
```
HF_token=your_hugging_face_token_here
```

## 🚀 Usage

### Phase 1: Data Ingestion & Tokenization

Run the main pipeline to ingest data and generate tokens:

```python
from data_ingestion import ingest_and_tokenize

# Process 5 samples from the MathStrike dataset
ingest_and_tokenize(
    dataset_name="Incinciblecolonel/MathStrike",
    data_dir="original_img",
    limit=5
)
```

This will:
- Download samples from the specified Hugging Face dataset
- Apply tokenization to each image
- Save results to `./tokenization_results/` directory

### Phase 2: Strikeout Detection Training

**Quick Demo (5,000 samples, 5 epochs)**
```bash
python train_demo.py --epochs 5 --num-train-samples 5000
```

**Full Training (97K samples, 30 epochs)**
```bash
python train_phase2_real_data.py \
    --data-dir ./data \
    --epochs 30 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --output-dir checkpoints_phase2_full
```

**Inference on New Components**
```bash
# Single image
python inference_strikeout_detector.py \
    --checkpoint checkpoints_phase2_full/checkpoint_best.pt \
    --image path/to/component.jpg

# Batch prediction
python inference_strikeout_detector.py \
    --checkpoint checkpoints_phase2_full/checkpoint_best.pt \
    --image-dir path/to/components/
```

See [PHASE2_GUIDE.py](PHASE2_GUIDE.py) for detailed documentation.

### Output Structure

Results are organized by sample:
```
tokenization_results/
├── sample_000000/
│   ├── tokens.json (token data)
│   └── crops/      (cropped regions)
├── sample_000001/
│   ├── tokens.json
│   └── crops/
...
```

## 📁 Project Structure

```
├── Phase 1: Tokenization
│   ├── data_ingestion.py           # Dataset loading & processing
│   ├── tokenization.py             # Tokenization pipeline
│   └── tokenization_results/       # Output directory
│
├── Phase 2: Strikeout Detection
│   ├── vit_strikeout_detector.py   # ViT model architecture & API
│   ├── data_loader_phase2.py       # Component extraction & dataset
│   ├── train_phase2_real_data.py   # Full training pipeline
│   ├── train_demo.py               # Quick demo training
│   ├── inference_strikeout_detector.py  # Inference on new data
│   └── PHASE2_GUIDE.py             # Comprehensive guide
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📚 Dependencies

**Core Libraries**
- **torch** — Deep learning framework
- **torchvision** — Computer vision utilities
- **transformers** — Hugging Face pre-trained models
- **einops** — Tensor operations

**Image Processing**
- **opencv-python** — Image processing & manipulation
- **Pillow** — Image I/O and manipulation
- **numpy** — Numerical computations
- **scipy** — Scientific algorithms (Delaunay triangulation)

**Utilities**
- **matplotlib** — Visualization
- **datasets** — Hugging Face datasets integration
- **tqdm** — Progress bars
- **python-dotenv** — Environment variable management

See `requirements.txt` for specific versions.

## 🔗 Dataset

This project uses the **MathStrike** dataset from Hugging Face:
- **Dataset**: `Incinciblecolonel/MathStrike`
- **Default Split**: `train`
- **Default Directory**: `original_img`

## 📑 Related Work

This project is inspired by and implements the methodology from:

**MathScrub: Single-Pipeline Strikeout Removal and LaTeX Generation for Handwritten Calculus Expressions**
- Authors: Sen Shen, Ning Liu, Lintao Song, Dongkun Han
- Published: IEEE Transactions on Learning Technologies, Vol. 19, 2026

## � Contributors

This project is a collaborative effort by students from the **Indian Institute of Information Technology, Lucknow**:

| Name | Roll No. | Role |
|------|----------|------|
| Urmi Kanrar | MSA25003 | Team Lead |
| Moumita Paul | MSA25014 | Team Member |
| Lipika Maji | MSD25011 | Team Member |
| Biplov Singh | MSD25010 | Team Member |

## 📝 License

See LICENSE file for details.

## 🛠️ Development Notes

- **Phase 1 (Token Construction)** has been completed successfully ✅
- **Phase 2 (Multimodal Deletion Classification)** is currently in development 🔄
- **Phase 3 & 4** are in planning stages 📋
- Features and APIs may change as the project evolves
- Contributions and feedback are welcome


**Last Updated**: April 2026  
**Current Phase**: Phase 1 Complete (Token Construction) | Next: Phase 2 (Multimodal Deletion Classification)
