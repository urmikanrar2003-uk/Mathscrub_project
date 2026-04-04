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

### Phase 2: Multimodal Deletion Classification (In Progress)

- Uses Vision Transformer (ViT) architecture for strikeout detection
- Three-channel input: grayscale image, component pixel area, and bounding box size
- Employs precise mathematical formulas for geometry normalization
- Identifies strikeouts in handwritten expressions for removal

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

### Data Ingestion & Tokenization

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
├── data_ingestion.py      # Dataset loading & processing
├── tokenization.py        # Tokenization pipeline implementation
├── requirements.txt       # Python dependencies
├── tokenization_results/  # Output directory
└── README.md             # This file
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
- **Phase 2 (Multimodal Deletion Classification)** is currently in development 🔄
- **Phase 3 & 4** are in planning stages 📋
- Features and APIs may change as the project evolves
- Contributions and feedback are welcome


**Last Updated**: April 2026  
**Current Phase**: Phase 1 Complete (Token Construction) | Next: Phase 2 (Multimodal Deletion Classification)
