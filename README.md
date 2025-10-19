# DeepFusion-DDG: Protein Stability Prediction

Deep learning models for predicting protein thermodynamic stability changes (ΔΔG) from single-point mutations using sequence and structure information.

## Project Overview

This project develops fusion models combining:
- **Sequence-based models**: ESM2 (Evolutionary Scale Modeling)
- **Structure-based models**: Graph Neural Networks (GNNs)
- **Baseline models**: Traditional ML approaches

**Dataset**: FireProtDB filtered dataset with 141 proteins, 4,291 single-point mutations

## Environment Setup

This project uses `uv` for fast, reproducible dependency management.

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Create and activate virtual environment
uv venv DeepFusion-DDG --python 3.11
source DeepFusion-DDG/bin/activate  # On macOS/Linux
# DeepFusion-DDG\Scripts\activate   # On Windows

# Install dependencies
uv pip install -e .
```

### Development Installation
```bash
uv pip install -e ".[dev]"
```

## Project Structure

```
BMI707FinalProject/
├── 707_Files_for_Colab/
│   ├── Data_Frames/           # Filtered FireProtDB datasets
│   ├── WT_PDBs/               # Wild-type protein structures
│   └── Data_To_Aggregate/     # Graph dataset PKL files
├── Official Model Notebooks/   # Trained models and notebooks
│   ├── BMI_707_Official_Aggregate_Data_05_08_2025.ipynb
│   ├── BMI_707_Official_ESMB_Model_From_Aggregated_05_08_2025.ipynb
│   ├── BMI_707_Official_GNN_Model_From_Aggregated_05_08_2025.ipynb
│   └── BMI_707_Official_Fusion_Model_From_Aggregated_05_08_2025.ipynb
├── Official Model Performance and Saved Weights/
│   ├── Baseline Models/
│   ├── Sequence Only Model/
│   ├── Structure Only Model/
│   └── Fusion Model/
└── Updates_for_MIT_Media_Lab_Collaboration/
    └── explore_filtered_fireprot_db.ipynb  # Dataset exploration
```

## Key Components

### Datasets
- **FireProtDB**: Curated protein stability database
- **ΔΔG values**: Ground truth thermodynamic stability changes
- **Protein structures**: PDB files for wild-type proteins

### Models
1. **Baseline Models**: Ridge, Random Forest, Gradient Boosting
2. **ESM2 Model**: Sequence-only transformer model
3. **GNN Model**: Structure-only graph neural network
4. **Fusion Model**: Combined sequence + structure approach

## Usage

```python
# Load the filtered dataset
import pandas as pd
df = pd.read_csv('707_Files_for_Colab/Data_Frames/df_fireprot_median_ddG_old.csv')

# Load trained models
import torch
model = torch.load('Official Model Performance and Saved Weights/Fusion Model/fusion_best_model.pt')
```

## Dependencies

Core dependencies:
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- BioPython 1.81+
- pandas, numpy, scikit-learn
- matplotlib, seaborn

See `pyproject.toml` for complete dependency list.

## Collaboration

For MIT Media Lab collaboration, see `Updates_for_MIT_Media_Lab_Collaboration/` directory.

## Authors

BMI 707 Final Project Team

## Date

October 2025
