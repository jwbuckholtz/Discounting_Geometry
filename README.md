# Decoding Delay Discounting (DecodingDD)

## 1. Project Overview

This project implements a comprehensive fMRI analysis pipeline to investigate the neural representations underlying delay discounting decisions. The primary goal is to leverage both standard univariate and advanced multivariate analysis techniques to decode and characterize neural patterns related to subjective value and choice.

The analysis is designed to be modular, reproducible, and scalable. It includes stages for:
- Behavioral modeling to estimate discount rates and trial-wise subjective values.
- First-level fMRI modeling using a standard GLM with parametric modulators.
- Group-level (second-level) analysis to identify group-average effects.
- First-level fMRI modeling using Least-Squares-Separate (LSS) for single-trial beta estimation.
- Multivariate Pattern Analysis (MVPA) for decoding.
- Representational Similarity Analysis (RSA).

The entire pipeline is designed to be executed on a high-performance computing (HPC) cluster using SLURM.

## 2. Repository Structure

```
/
├── config/               # Project and analysis parameter configuration
│   └── project_config.yaml
├── derivatives/          # All analysis outputs (gitignored)
├── scripts/              # All analysis and utility scripts
│   ├── behavioral/
│   ├── modeling/
│   ├── group_level/
│   ├── mvpa/
│   └── rsa/
├── slurm/                # SLURM job submission scripts
├── tests/                # Automated test suite for the analysis scripts
├── .gitignore
└── pyproject.toml        # Project metadata and Python dependencies
```

## 3. Setup on an HPC (SLURM)

Follow these steps on the HPC login node to set up the project.

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd DecodingDD
    ```

2.  **Load the Correct Python Module**:
    Your project requires Python 3.9+. Load the appropriate module before proceeding.
    ```bash
    ml python/3.9
    ```

3.  **Create and Activate Virtual Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

4.  **Install Dependencies**:
    First, upgrade `pip` to a modern version, then install the project and its dependencies from `pyproject.toml`.
    ```bash
    pip install --upgrade pip
    pip install -e .
    ```

## 4. Configuration

All paths and analysis parameters are managed through a central configuration file: `config/project_config.yaml`.

**Crucially, before running any analyses, you must edit this file** to provide the correct absolute paths for your cluster environment in the `hpc` section.

## 5. HPC Analysis Workflow

The pipeline is designed to be run as a sequence of discrete stages on a SLURM cluster. This provides clear checkpoints and makes debugging easier.

**Before you begin**: Grant execute permissions to all submission scripts:
```bash
chmod +x slurm/*.sh slurm/*.sbatch
```

### Step 1: Behavioral Analysis
This step runs the behavioral modeling for all subjects, generating the trial-wise subjective value files required by all fMRI models.
```bash
sbatch slurm/submit_behavioral_analysis.sbatch
```

### Step 2: First-Level Standard GLMs
This submits a separate job for each subject to run the standard first-level GLM, generating contrast maps for each predictor.
```bash
./slurm/submit_all_standard_glms.sh
```

### Step 3: Group-Level Univariate Models
*This step should only be run after all jobs from Step 2 have completed successfully.*
This runs a single job that performs a group-level (second-level) analysis for all contrasts, identifying significant group-average effects.
```bash
sbatch slurm/submit_group_glms.sbatch
```

### Step 4: First-Level LSS Models
This submits a separate job for each subject to generate single-trial beta-maps using the Least-Squares-Separate (LSS) method. These are required for MVPA and RSA.
```bash
./slurm/submit_all_lss_models.sh
```

### Step 5: MVPA and RSA
*These steps should only be run after all jobs from Step 4 have completed successfully.*
These scripts submit jobs for each subject to run the multivariate analyses.
```bash
# To run MVPA (decoding)
./slurm/submit_all_decoding.sh

# To run RSA
./slurm/submit_all_rsa.sh
```

### Step 6: Visualizations
This step will generate plots and figures from the results of the previous stages.
```bash
./slurm/submit_all_visualizations.sh
```

---
You can monitor the progress of all submitted jobs using `squeue -u $USER`. Logs for each job are saved to the `logs/` directory.
