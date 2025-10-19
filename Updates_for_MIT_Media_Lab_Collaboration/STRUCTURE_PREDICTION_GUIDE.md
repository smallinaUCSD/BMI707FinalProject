# Structure Prediction Guide - Building Intuition

**Purpose:** Understand how to get protein structures for wild-types and mutants  
**Date:** October 17, 2025  
**Audience:** Researchers new to protein structure prediction

---

## The Core Question: Do We Predict or Download?

### For **Wild-Type (WT) Proteins**: Download First, Predict if Needed

**AlphaFold Database is Your First Stop**

Think of AlphaFold Database like Google Images, but for protein structures:
- **Pre-computed:** Meta/DeepMind already predicted structures for ~200M proteins
- **Search by UniProt ID:** Yes! This is the primary way to search
- **Free & Instant:** No computation needed on your end
- **URL Format:** `https://alphafold.ebi.ac.uk/files/AF-{UNIPROT_ID}-F1-model_v4.pdb`

**How it works:**
1. You have UniProt ID (e.g., P00720)
2. Make HTTP request to AlphaFold DB API
3. If exists: Download PDB file (instant)
4. If not exists: Need to predict from scratch

**For our missing proteins:**
- P00720 (Endolysin) - Check AlphaFold DB first
- P69543 (DNA-Binding protein G5P) - Check AlphaFold DB first

**No API key needed!** Just HTTP GET requests.

---

### For **Mutant Proteins**: Always Predict (Not in Any Database)

**Why mutants aren't in databases:**
- Databases have wild-type sequences from genomes
- Your mutants are synthetic variants (T349F, etc.)
- Each mutation = unique sequence = needs individual prediction
- No pre-computed database for every possible mutation

**You must predict from scratch.**

---

## Prediction Tools Landscape

### 1. **ESMFold** ⭐ RECOMMENDED FOR YOUR CASE

**What it is:**
- Meta AI's protein folding model (2022)
- Based on protein language models (no need for MSA like AlphaFold2)
- Optimized for speed

**Speed:** 
- 1-2 seconds per protein
- 72 mutants = ~2-3 minutes total

**Accuracy:**
- Near-AlphaFold2 quality (~95% as good)
- Excellent for single-point mutations
- Validated on millions of structures

**How to use:**

**Option A: API (Simplest)**
```python
import requests

def predict_with_esmfold(sequence):
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    response = requests.post(url, data=sequence)
    return response.text  # PDB file content

# That's it! No API key, no setup
```

**Option B: Local Install (For offline/batching)**
```bash
pip install fair-esm
```

**Why ESMFold for you:**
- ✓ Your 72 mutations in minutes vs hours
- ✓ No GPU required (API handles it)
- ✓ Quality sufficient for ΔΔG correlation studies
- ✓ Free and stable API

---

### 2. **AlphaFold2** - The Gold Standard (But Slower)

**What it is:**
- DeepMind's original breakthrough (2020)
- Uses MSA (Multiple Sequence Alignment) + deep learning
- Highest accuracy overall

**Speed:**
- 5-10 minutes per protein on GPU
- 72 mutants = 6-12 hours total

**Accuracy:**
- Best available (~98% of experimental structures)
- Especially good for complex folds
- Includes confidence scores (pLDDT)

**How to use:**

**Option A: Google Colab (Easiest)**
- Go to ColabFold notebook: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb
- Paste sequence
- Click run
- Free GPU provided by Google
- Limitation: ~12 hours/day max

**Option B: Local Install (Complex)**
- Requires NVIDIA GPU (8GB+ VRAM)
- Docker recommended
- Full setup takes ~2-3 hours first time
- Good if you do this regularly

**When to use AlphaFold2:**
- Critical structures (e.g., near active sites)
- Need absolute highest accuracy
- Publication-quality structures
- Validation of ESMFold results (spot checks)

---

### 3. **Boltz-2** - The New Kid (October 2024)

**What it is:**
- MIT/Harvard's new model
- Claims to match/exceed AlphaFold3
- Designed for protein complexes

**Speed:**
- 2-5 minutes per structure

**Accuracy:**
- Claims very high, but:
  - Only released Oct 2024
  - Not extensively tested yet
  - Limited benchmarks on mutations

**How to use:**
```bash
pip install boltz
```

**Should you use it?**
- **Not yet** for your project
- Too new, not proven for mutations
- Stick with ESMFold/AF2 (battle-tested)
- Revisit in 6-12 months when validated

---

### 4. **Other Options** (For Completeness)

**RosettaFold:**
- UW/IPDB alternative to AlphaFold2
- Similar accuracy, similar speed
- More complex setup
- Use case: Already using Rosetta suite

**ESMFold Atlas:**
- Pre-computed ESMFold structures (700M+ sequences)
- Like AlphaFold DB but for metagenomic sequences
- Unlikely to have your specific proteins
- Check if interested: https://esmatlas.com/

---

## Practical Workflow for Your Project

### Phase 1: Get Missing WT Structures (2 proteins)

```python
# For P00720 and P69543

# Step 1: Check AlphaFold Database
url = f"https://alphafold.ebi.ac.uk/files/AF-P00720-F1-model_v4.pdb"
response = requests.head(url)

if response.status_code == 200:
    # Download it!
    pdb_data = requests.get(url).text
    save_to_file(pdb_data, "P00720.pdb")
else:
    # Not in database, predict with ESMFold
    sequence = get_sequence("P00720")  # from your CSV
    pdb_data = predict_with_esmfold(sequence)
    save_to_file(pdb_data, "P00720.pdb")
```

**Time:** ~2 minutes (instant if in AlphaFold DB, ~2 sec if predicting)

---

### Phase 2: Generate All Mutant Structures (72 mutants)

```python
# For each mutant in your CSV

for idx, row in mutants_df.iterrows():
    # Get mutant sequence
    mutant_seq = row['mt_sequence']
    
    # Predict structure
    pdb_data = predict_with_esmfold(mutant_seq)
    
    # Save with naming convention
    filename = f"{row['uniprot_id']}_{row['mutation_notation']}.pdb"
    save_to_file(pdb_data, filename)
    
    # Small delay to avoid rate limiting
    time.sleep(0.5)
```

**Time:** ~3-5 minutes for all 72

---

### Phase 3: Quality Control

**Check pLDDT scores:**
- ESMFold includes confidence scores in B-factor column
- pLDDT > 70 = good confidence
- pLDDT > 90 = very high confidence

**Validate key mutations:**
- Pick 5-10 critical mutants (e.g., in catalytic pockets)
- Re-predict with AlphaFold2 via Colab
- Compare RMSD between ESMFold and AF2
- If RMSD < 2Å, ESMFold is good enough

**Visual inspection:**
- Load in PyMOL or Mol*
- Check mutation is at correct position
- Verify structure looks reasonable (no clashes, proper folding)

---

## Key Concepts Explained

### 1. **What's an API call vs. local prediction?**

**API Call (ESMFold API):**
- Your code sends sequence to Meta's servers
- Their GPUs do the work
- You get back PDB file
- Like using a website but via code
- **Pros:** No setup, no GPU needed, fast
- **Cons:** Need internet, rate limits possible

**Local Prediction:**
- You install software on your computer
- Your GPU does the work
- No internet needed after setup
- **Pros:** Unlimited runs, faster for batches, offline
- **Cons:** Requires GPU, complex setup

### 2. **What's a UniProt ID?**

- Universal Protein Resource identifier
- Like a social security number for proteins
- Format: P00720, P69543, etc.
- Stable across databases
- Links to sequence, function, structure data

### 3. **Why AlphaFold DB search works by UniProt:**

- AlphaFold DB predicted structures for UniProt reference proteomes
- Each prediction = one UniProt entry
- URL scheme: `/AF-{UNIPROT}-F1-model_v4.pdb`
- "F1" = Fragment 1 (full-length protein)
- "v4" = Version 4 (latest model)

### 4. **What's a PDB file?**

- Protein Data Bank format
- Text file with 3D coordinates of atoms
- Lines start with "ATOM" for each atom
- Includes x, y, z positions
- Standard format readable by all structure tools

---

## Decision Tree for Your Project

```
Need structure?
│
├─ Wild-type protein?
│  ├─ Check AlphaFold DB by UniProt ID
│  │  ├─ Found? → Download (instant)
│  │  └─ Not found? → Predict with ESMFold (2 sec)
│  
└─ Mutant protein?
   └─ Predict with ESMFold (2 sec each)
      └─ Optional: Validate critical ones with AF2 on Colab
```

---

## Recommendations Summary

### For Your 72 Mutants:

1. **Primary method: ESMFold API**
   - Fast (3 minutes total)
   - Free, no setup
   - Quality sufficient for ΔΔG studies

2. **Validation: AlphaFold2 (Colab) for 5-10 key mutants**
   - Near catalytic sites
   - Largest |ΔΔG| values
   - One per protein as spot check

3. **Skip Boltz-2 for now**
   - Too new, unproven
   - Revisit in future work

### For Missing WT Structures:

1. **Check AlphaFold DB first** (instant)
2. **If not found:** ESMFold API (2 sec each)
3. **Don't overthink it:** Both are high quality

---

## Code Template for Complete Workflow

```python
import requests
import pandas as pd
from pathlib import Path
import time

# Load your mutants
df = pd.read_csv("mit_media_lab_selected_mutants.csv")

# Create output directory
output_dir = Path("predicted_structures")
output_dir.mkdir(exist_ok=True)

# Function to predict
def predict_structure(sequence, save_path):
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    response = requests.post(url, data=sequence, timeout=60)
    
    if response.status_code == 200:
        with open(save_path, 'w') as f:
            f.write(response.text)
        return True
    return False

# Generate all mutants
for idx, row in df.iterrows():
    filename = f"{row['uniprot_id']}_{row['mutation_notation']}.pdb"
    save_path = output_dir / filename
    
    print(f"Predicting {filename}...")
    success = predict_structure(row['mt_sequence'], save_path)
    
    if success:
        print(f"  ✓ Saved to {save_path}")
    else:
        print(f"  ✗ Failed")
    
    time.sleep(0.5)  # Be nice to the API

print(f"\nDone! Check {output_dir}/")
```

---

## Resources

- **AlphaFold DB:** https://alphafold.ebi.ac.uk/
- **ESMFold API:** https://esmatlas.com/about#api
- **ESMFold Paper:** https://www.science.org/doi/10.1126/science.ade2574
- **ColabFold (AF2):** https://github.com/sokrypton/ColabFold
- **Boltz-2 (new):** https://github.com/jwohlwend/boltz

---

**Bottom Line:** 
- Use ESMFold API for all 72 mutants (fast, easy, good quality)
- Check AlphaFold DB for missing WTs (instant if available)
- Validate a few with AF2 if you want peace of mind
- Don't overthink it - ESMFold is proven for this use case

