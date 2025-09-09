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

The pipeline is designed to be run as a sequence of discrete stages on a SLURM cluster. We use modern SLURM **job arrays** for the subject-level analyses, which is highly efficient for managing large batches of jobs.

**Before you begin**: You must first `git pull` on Sherlock to get the latest versions of these scripts. Then, ensure the SLURM scripts are executable:
```bash
chmod +x slurm/*.sbatch
```

### Step 1: First-Level Standard GLMs
This submits a **single job array** that processes a specific list of subjects defined within the script. This script generates the first-level contrast maps needed for the group analysis.

```bash
sbatch slurm/submit_glm_batch.sbatch
```

### Step 2: First-Level LSS Models
*This can be run in parallel with Step 1.*
This submits another **job array** to generate single-trial beta-maps using the Least-Squares-Separate (LSS) method. This is a very computationally intensive step and may take several hours per subject. This script will automatically find and process **all** subjects in your `derivatives/behavioral` directory.

```bash
sbatch slurm/submit_lss_batch.sbatch
```

### Step 3: Group-Level Univariate Models
*This step should only be run after all jobs from Step 1 have completed successfully.*
This runs a single job that performs a group-level (second-level) analysis for all contrasts defined in `project_config.yaml`, identifying significant group-average effects across the cohort.

```bash
sbatch slurm/submit_group_glms.sbatch
```

### Step 4 (Optional): Behavioral Data Summary & QA
We have created a powerful script to summarize behavioral data, run key statistical models (mixed-effects models), and generate diagnostic plots. This is an excellent tool for quality assurance and for understanding your behavioral data in depth.

This script can be run locally (if OAK is mounted) or on an HPC login node, as it is not computationally intensive.

```bash
# To run using local paths from the config file
python scripts/behavioral/summarize_behavioral_data.py --env local

# To run using HPC (OAK) paths from the config file
python scripts/behavioral/summarize_behavioral_data.py --env hpc
```
The script will print all statistical model summaries to the terminal and save all output tables and plots to `derivatives/behavioral/summaries/`.

---
You can monitor the progress of all submitted jobs using `squeue -u $USER`. Logs for each job and task array are saved to the `logs/` directory.
