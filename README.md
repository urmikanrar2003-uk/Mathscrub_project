# Mathscrub Project

Single-pipeline strikeout removal and LaTeX generation for handwritten calculus expressions.

## 📌 Status

🚧 **Development Phase** — Features and APIs may change

### Project Phases

| Phase | Module | Description | Status |
|-------|--------|-------------|--------|
| **Phase 1** | Token Construction | Delaunay triangulation and Union-Find grouping for semantic symbol clustering | ✅ **COMPLETE** |
| **Phase 2** | Multimodal Deletion Classification | Vision Transformer (ViT) for strikeout detection and classification | ✅ **COMPLETE** |
| **Phase 3** | Geometry-Aware Inpainting | Image restoration using Navier-Stokes interpolation after strikeout removal | ✅ **COMPLETE** |
| **Phase 4** | LaTeX Generation | Vision-Language Model (Qwen2.5-VL-7B/32B) transcription with PEFT | 📋 Planned |

## 📖 Overview

Mathscrub is a recreation of the research paper **"MathScrub: Single-Pipeline Strikeout Removal and LaTeX Generation for Handwritten Calculus Expressions"** (Sen Shen et al., IEEE Transactions on Learning Technologies, 2026).

This project is designed to extract and tokenize mathematical symbols and structures from handwritten document images. It ingests data from Hugging Face datasets and applies a sophisticated four-module pipeline to process handwritten calculus expressions, ultimately generating LaTeX code.

### Key Features

- **Data Ingestion**: Automatically loads image datasets from Hugging Face
- **Image Tokenization**: Converts mathematical images into semantic tokens using Delaunay triangulation and Union-Find clustering *(Phase 1 Complete)*
- **Structured Output**: Generates tokens and cropped regions for further analysis
- **Strikeout Detection**: Uses Vision Transformers for deletion classification *(Phase 2 Complete)*
- **Image Restoration**: Geometry-aware inpainting using Navier-Stokes and boundary feathering *(Phase 3 Complete)*
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

### Phase 2: Multimodal Deletion Classification ✅ (COMPLETE)

- Uses an Early-Fusion Vision Transformer (ViT) architecture natively mapping 3 channels (Grayscale Image Crop + Area Geometry Scalar + Maximum Bounding Box Scalar).
- Computes empirical geometry statistics natively over tokenization arrays padding out crops to perfect structural squares (`224x224`).
- Instantly predicts deletions traversing `tokens.json` directly within the streaming dataset loop natively.

### Phase 3: Geometry-Aware Inpainting ✅ (COMPLETE)

- **Navier-Stokes Interpolation** — Applied for boundary-aware restoration of small symbol gaps.
- **Boundary-Aware Feathering** — Implements MathScrub Eq. 10 for seamless repasting of larger strikeout regions.
- **Area Thresholding** — Uses precise size thresholds ($T=15316$) to separate symbol gaps from massive occlusions.
- **Unified Pipeline** — Full-dataset restoration script (`restore_dataset.py`) processes images end-to-end and saves cleaned images directly to the D: drive.

### Phase 4: LaTeX Generation (Planned)

- **Vision-Language Model**: Qwen2.5-VL (7B or 32B version) for expression transcription.
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

### Full Dataset Restoration (Phases 1-3)

To process the entire MathStrike dataset and save cleaned images for fine-tuning, run the unified pipeline:

```powershell
# Optimized for D: drive and GPU acceleration
python restore_dataset.py --output_dir "D:/MathScrub_Restored"
```

**Pipeline Workflow**:
1. **Streaming** — Images are streamed from Hugging Face in memory.
2. **In-Memory Processing** — Tokenization and ViT classification are done without writing temporary crops to disk.
3. **Automatic Cleanup** — Intermediate data is discarded after each sample to save space.
4. **Resumption Support** — If stopped, the script automatically picks up from the last processed sample.

### Output Structure

Restored images are saved to your specified output directory (e.g., D: drive):
```
D:/MathScrub_Restored/
├── progress.txt             # Tracks completed sample IDs
├── failed_samples.txt       # Logs IDs of corrupted/failed images
├── sample_000000/
│   ├── restored_img_final.png  # Cleaned math image (ready for VLM)
│   └── meta.json               # Slim metadata (ID, shape)
├── sample_000001/
│   ├── restored_img_final.png
│   └── meta.json
...
```

## 📁 Project Structure

```
├── restore_dataset.py        # Unified Phase 1-3 full-dataset pipeline
├── geometry_inpainting.py     # Phase 3: Restoration logic & main
├── tokenization.py           # Phase 1: Semantic token construction
├── vit_strikeout_detector.py # Phase 2: Multimodal Early-Fusion ViT script
├── data_ingestion.py         # Original ingestion script for Phase 1 testing
├── train_vit.py              # ViT training script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 📚 Dependencies

- **opencv-python** — Image processing
- **numpy** — Numerical computations
- **scipy** — Scientific algorithms (Delaunay triangulation)
- **matplotlib** — Visualization
- **Pillow** — Image manipulation
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
- **Phase 2 (Multimodal Deletion Classification)** has been completed successfully ✅
- **Phase 3 (Geometry-Aware Inpainting)** has been completed successfully ✅
- **Phase 4 (LaTeX Generation)** is in the planning stage (Next Step) 📋
- The pipeline is now ready for full-scale data cleaning on the MathStrike dataset.

**Last Updated**: May 2026  
**Current Phase**: Phase 3 Complete (Restoration) | Next: Phase 4 (LaTeX Transcription)
