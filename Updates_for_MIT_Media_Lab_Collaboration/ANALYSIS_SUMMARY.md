# FireProtDB Dataset Analysis - MIT Media Lab Collaboration

**Date:** October 17, 2025  
**Analyst:** Shyam Chandra  
**Purpose:** Select representative protein mutants for MIT Media Lab collaboration

---

## Executive Summary

Successfully identified and curated **72 mutants** across **12 proteins** from the filtered FireProtDB dataset, achieving perfect 3:3 balance of stabilizing and destabilizing mutations for each protein.

---

## Dataset Overview

### Source Dataset
- **File:** `df_fireprot_median_ddG_old.csv`
- **Total Mutations:** 5,086
- **Unique Proteins:** 174 (by UniProt ID)
- **Note:** Discrepancy from reported 4,291 mutations likely due to intermediate filtering steps or structure generation failures

### ΔΔG Distribution (Threshold: ±0.5 kcal/mol)
- **Destabilizing (ddG > 0.5):** 2,740 (53.9%)
- **Near-zero (-0.5 ≤ ddG ≤ 0.5):** 1,641 (32.3%)
- **Stabilizing (ddG < -0.5):** 705 (13.9%)

**Key Insight:** Stabilizing mutations are scarce (~14%), making balanced selection challenging but achievable for top proteins.

---

## Top 12 Most Representative Proteins

| Rank | UniProt ID | Protein Name | Mutations | WT Structure |
|------|------------|--------------|-----------|--------------|
| 1 | P06654 | Immunoglobulin G-binding protein G | 941 | ✓ |
| 2 | P00644 | Thermonuclease | 579 | ✓ |
| 3 | P00720 | Endolysin | 308 | ✗ |
| 4 | P00648 | Ribonuclease | 176 | ✓ |
| 5 | P61626 | Lysozyme C | 133 | ✓ |
| 6 | P07751 | Spectrin alpha chain, non-erythrocytic 1 | 119 | ✓ |
| 7 | P0ABQ4 | Dihydrofolate reductase | 111 | ✓ |
| 8 | P69543 | DNA-Binding protein G5P | 92 | ✗ |
| 9 | P01053 | Subtilisin-chymotrypsin inhibitor-2A | 86 | ✓ |
| 10 | P0A7Y4 | Ribonuclease HI | 79 | ✓ |
| 11 | P02185 | Myoglobin | 76 | ✓ |
| 12 | P04637 | Cellular tumor antigen p53 | 72 | ✓ |

**Total mutations in top 12:** 2,772 (54.5% of entire dataset)

---

## Selection Strategy

### Goals
- **6 mutants per protein** (72 total)
- **3 stabilizing + 3 destabilizing** per protein
- Prioritize mutants with **larger |ΔΔG|** values (more extreme effects)

### Results
✓ **Perfect balance achieved:** 36 stabilizing, 36 destabilizing across all 12 proteins  
✓ **ΔΔG range:** -4.60 to +9.70 kcal/mol  
✓ **No near-zero mutations included** (clean separation)

---

## Structure Availability

### Wild-Type Structures
- **Available:** 10/12 proteins (83%)
- **Need Generation:** 2/12 proteins (17%)
  - P00720 (Endolysin)
  - P69543 (DNA-Binding protein G5P)

### Mutant Structures
- **All 72 mutants require structure generation**
- Recommended approach: AlphaFold2 or ESMFold for single-point mutations

---

## Deliverables

### 1. CSV Export
**File:** `mit_media_lab_selected_mutants.csv`

**Columns:**
- `rank`, `protein_name`, `uniprot_id`, `pdb_id`, `chain`
- `mutation_notation` (e.g., "T349F"), `position`, `wild_type`, `mutation`
- `ddG`, `stability_class`
- `wt_structure_path`, `mutant_structure_status`
- `experiment_id`, `sequence`, `mt_sequence`
- `is_in_catalytic_pocket`, `is_essential`

### 2. Jupyter Notebook
**File:** `explore_filtered_fireprot_db.ipynb`

**Contents:**
- Step 0: Dataset loading and validation
- Step 1: ΔΔG distribution analysis
- Step 2: Top 12 protein identification
- Step 3: Structure availability check
- Step 4: Intelligent mutant selection
- Step 5: Summary visualizations
- Step 6: Detailed mutant listing
- Step 7: CSV export

---

## Next Steps

### Phase 1: Structure Generation (Priority)
1. **WT Structures:** Generate for P00720 and P69543
2. **Mutant Structures:** Generate all 72 using:
   - AlphaFold2 (gold standard, slower)
   - ESMFold (faster alternative)
   - Verify mutation at correct position

### Phase 2: Quality Control
1. Validate WT structures align with PDB chains
2. Check structural integrity post-mutation
3. Verify mutation sites are accessible
4. Compare predicted vs experimental structures (where available)

### Phase 3: Final Packaging
1. Create comprehensive PKL file with:
   - Protein metadata
   - Sequences (WT and mutant)
   - Structures (PDB format or tensors)
   - ΔΔG values and classifications
   - Experiment IDs and references
2. Generate final documentation
3. Prepare for handoff to Prof and Allan

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Proteins Selected | 12 |
| Mutants per Protein | 6 |
| Total Mutants | 72 |
| Stabilizing Mutants | 36 (50%) |
| Destabilizing Mutants | 36 (50%) |
| WT Structures Available | 10/12 (83%) |
| Mutant Structures Needed | 72 (100%) |
| ΔΔG Range | -4.60 to +9.70 kcal/mol |

---

## File Locations

```
Updates_for_MIT_Media_Lab_Collaboration/
├── explore_filtered_fireprot_db.ipynb          # Main analysis notebook
├── mit_media_lab_selected_mutants.csv          # Curated dataset (72 mutants)
└── ANALYSIS_SUMMARY.md                         # This document

707_Files_for_Colab/
├── Data_Frames/
│   └── df_fireprot_median_ddG_old.csv          # Source dataset (5,086 mutations)
└── WT_PDBs/
    ├── P06654.pdb                               # Available WT structures
    ├── P00644.pdb
    └── ... (10 total)
```

---

## Notes

1. **Dataset Discrepancy:** The source CSV has 5,086 mutations vs. reported 4,291. This may be due to:
   - Intermediate filtering steps not captured in this CSV
   - Structure generation failures (see `df_with_skipped_flag.csv`)
   - Recommend investigating if critical

2. **Stabilizing Mutations Scarcity:** Only 13.9% of dataset is stabilizing (ddG < -0.5). Successfully found 3+ for each top protein, but this is a limiting factor for expansion.

3. **Structure Generation Tools:** 
   - AlphaFold2: Most accurate, ~10min per structure (Google Colab)
   - ESMFold: Faster (~1min), slightly less accurate
   - Both suitable for single-point mutations

4. **PDB File Sizes:** Average ~150KB per structure. Total for 72 mutants + 2 missing WTs = ~11MB (very manageable).

---

## Contact

For questions or clarifications, contact Shyam Chandra.

**Generated:** October 17, 2025
