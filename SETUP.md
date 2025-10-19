# Quick Start Guide - DeepFusion-DDG Environment

## For Collaborators

### First Time Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd BMI707FinalProject

# Install uv (if not already installed)
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create the environment
uv venv DeepFusion-DDG --python 3.11

# Activate the environment
source DeepFusion-DDG/bin/activate  # macOS/Linux
# DeepFusion-DDG\Scripts\activate   # Windows

# Install all dependencies
uv pip install -e .
```

### Daily Usage

```bash
# Activate environment
source DeepFusion-DDG/bin/activate  # macOS/Linux

# Deactivate when done
deactivate
```

### For Jupyter Notebooks

After activating the environment, VS Code should automatically detect it.
If not, click the kernel selector in the top-right of the notebook and choose:
**Python 3.11.13 ('DeepFusion-DDG')**

### Adding New Dependencies

```bash
# Activate environment first
source DeepFusion-DDG/bin/activate

# Install new package
uv pip install <package-name>

# Update pyproject.toml
# Add the package to the dependencies list in pyproject.toml
```

### Syncing Dependencies

When someone adds new dependencies:

```bash
# Pull latest changes
git pull

# Sync your environment
uv pip install -e .
```

## Environment Info

- **Name**: DeepFusion-DDG
- **Python Version**: 3.11.13
- **Location**: `BMI707FinalProject/DeepFusion-DDG/`
- **Key Packages**: PyTorch 2.9, PyTorch Geometric 2.7, BioPython 1.85

## Troubleshooting

### "uv command not found"
Install uv: https://github.com/astral-sh/uv#installation

### VS Code not detecting kernel
1. Open Command Palette (Cmd+Shift+P / Ctrl+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose the DeepFusion-DDG environment

### Package import errors
```bash
# Reinstall dependencies
source DeepFusion-DDG/bin/activate
uv pip install -e . --force-reinstall
```

## Why uv?

- ⚡ **10-100x faster** than pip
- 🔒 **Reproducible** - exact dependency resolution
- 🎯 **Simple** - single tool for environments and packages
- 🤝 **Collaborative** - easy to share and sync

---
Last updated: October 17, 2025
