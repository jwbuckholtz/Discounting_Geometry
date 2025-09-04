# Decoding Delay Discounting (DecodingDD)

## 1. Project Overview

This project implements a comprehensive fMRI analysis pipeline to investigate the neural representations underlying delay discounting decisions. The primary goal is to leverage multivariate pattern analysis (MVPA) and representational similarity analysis (RSA) to decode and characterize the geometry of neural patterns related to subjective value and choice.

The analysis is designed to be modular, reproducible, and scalable, adhering to the "Better code, better science" principles. It includes stages for:
- Behavioral modeling to estimate discount rates and trial-wise subjective values.
- Single-trial fMRI modeling using Least-Squares-Separate (LSS) to generate beta maps for each decision.
- MVPA (decoding) to test for brain-based classification of key task variables.
- RSA to compare the representational geometry of neural patterns to theoretical models of choice and value.
- Visualization of neural embeddings using techniques like MDS and UMAP.

The entire pipeline is designed to be executed on both local machines for testing and on a high-performance computing (HPC) cluster using SLURM for large-scale analysis.

## 2. Repository Structure

This repository is organized to clearly separate code, data, configuration, and results.

```
/
├── config/               # Project configuration files
│   └── project_config.yaml
├── data/                 # Raw BIDS data (gitignored, should be a symlink or copy)
├── derivatives/          # All analysis outputs (gitignored)
│   ├── behavioral/
│   ├── fmriprep/
│   ├── lss_betas/
│   ├── mvpa/
│   ├── rsa/
│   └── visualization/
├── scripts/              # All analysis and utility scripts
│   ├── behavioral/
│   ├── modeling/
│   ├── mvpa/
│   ├── rsa/
│   └── visualization/
├── slurm/                # SLURM job submission scripts and logs
│   ├── submit_all_...sh
│   └── submit_...sbatch
├── .gitignore            # Specifies files for Git to ignore
└── pyproject.toml        # Project metadata and Python dependencies for `uv`
```

- **`config/`**: Contains `project_config.yaml` for specifying file paths for different execution environments (e.g., `local`, `hpc`).
- **`data/`**: Intended for raw BIDS-formatted data. This directory is included in `.gitignore` to prevent large data files from being committed to the repository.
- **`derivatives/`**: The primary output directory for all generated files, including behavioral results, beta maps, and analysis outputs. Also gitignored.
- **`scripts/`**: Contains all Python scripts, organized into subdirectories based on their analysis stage (e.g., `behavioral`, `modeling`).
- **`slurm/`**: Holds SLURM `.sbatch` templates and the `.sh` wrapper scripts used to submit jobs to the HPC cluster. Log files from SLURM jobs are also saved here.
- **`pyproject.toml`**: The definitive file for project dependencies, managed by the `uv` package manager.

## 3. Setup and Installation

To get started with this project, you will need `git` and `uv` installed on your system.

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone <repository_url>
cd DecodingDD
```

### 2. Create the Python Environment

This project uses `uv` for fast and reproducible Python environment management.

First, ensure you have a compatible version of Python installed (Python 3.9 is recommended). Then, to create and activate the project's virtual environment, run the following commands from the root of the project directory:

```bash
# Create the virtual environment using Python 3.9
python3.9 -m venv .venv

# Activate the newly created environment (on macOS/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies

Once the environment is activated, install all the required Python packages using `uv` and the `pyproject.toml` file. This command ensures you have the exact versions of all dependencies needed to run the code.

```bash
# First, install uv into the new environment
pip install uv

# Then, sync the environment with the project dependencies
uv pip sync pyproject.toml
```

After these steps, your environment is fully configured and ready for running the analysis pipeline.

## 4. Configuration

All paths for data input and result output are managed through a central configuration file: `config/project_config.yaml`. This approach follows the "Separation of Configuration from Code" principle, allowing you to adapt the pipeline to different computing environments without ever modifying the analysis scripts themselves.

The `project_config.yaml` file is structured with different sections for each environment, typically `local` for your personal computer and `hpc` for a cluster.

### Example `project_config.yaml`

```yaml
local:
  bids_dir: /path/to/your/local/data/bids
  derivatives_dir: /path/to/your/local/derivatives
  fmriprep_dir: /path/to/your/local/derivatives/fmriprep
  onsets_dir: /path/to/your/local/onsets

hpc:
  bids_dir: /path/on/hpc/to/data/bids
  derivatives_dir: /path/on/hpc/to/derivatives
  fmriprep_dir: /path/on/hpc/to/derivatives/fmriprep
  onsets_dir: /path/on/hpc/to/onsets
```

### Path Descriptions

-   **`bids_dir`**: The absolute path to the root of your BIDS-formatted dataset. This directory should contain your `sub-<ID>` folders.
-   **`derivatives_dir`**: The absolute path to the root directory where all analysis outputs will be saved. The scripts will create subdirectories within this location for each analysis stage (e.g., `lss_betas`, `rsa`).
-   **`fmriprep_dir`**: The absolute path to the output of your fMRIPrep preprocessing pipeline.
-   **`onsets_dir`**: The absolute path to the directory containing your trial onset files (e.g., `_events.tsv`), if they are stored separately from the BIDS data.

Before running any analysis, you must edit this file and replace the placeholder paths with the correct absolute paths for your system(s). When you run a script, you can specify which environment to use with the `--env` command-line argument (e.g., `--env local` or `--env hpc`).

## 5. Automated Testing with `pytest`

This project includes a comprehensive suite of automated tests to ensure the correctness and reliability of the core analysis functions. We use the `pytest` framework, which allows for fast, scalable, and easy-to-write tests. Adhering to the "Better code, better science" principles, these tests are a critical component for ensuring scientific validity.

Our testing strategy includes:
- **Unit Tests** for pure, algorithmic functions (e.g., `hyperbolic_discount`).
- **Integration Tests** for more complex functions that interact with neuroimaging data structures. These tests use fixtures to generate small, temporary, "fake" datasets on the fly, ensuring that the tests are fast, self-contained, and do not depend on the full dataset.

The test suite currently provides coverage for all major analysis stages:
- **`tests/test_behavioral.py`**: Validates the hyperbolic discounting model, choice probability functions, and the parameter recovery of the fitting algorithm.
- **`tests/test_modeling.py`**: Validates the LSS beta-series modeling pipeline, ensuring it correctly processes inputs and produces valid NIfTI image outputs.
- **`tests/test_mvpa.py`**: Validates the decoding analysis for both classification and regression tasks.
- **`tests/test_rsa.py`**: Validates the RSA pipeline by testing its ability to correctly recover a known, artificially embedded geometric structure from fake neural data.

### 1. Install Testing Dependencies
To run the tests, you first need to install the project in "editable" mode along with the testing extras:
```bash
uv pip install -e ".[test]"
```

### 2. Run the Test Suite
Once installed, you can run all tests by simply executing `pytest` from the root of the project directory:
```bash
pytest
```
The tests are located in the `tests/` directory and automatically check the validity of functions in the `scripts/` directory.

## 6. Running the Analysis Pipeline

The analysis pipeline is designed to be run sequentially. The output of one script often serves as the input for a later script. All commands should be run from the root of the project directory.

### Step 1: Behavioral Analysis

This step calculates the hyperbolic discount model parameters (`k` and `tau`) for each subject and computes the trial-wise subjective values (`SVchosen`, `SVdiff`, etc.).

**Command:**
```bash
# Run for a single subject
./.venv/bin/python scripts/behavioral/calculate_discount_rates.py --subjects sub-s061

# Run for multiple subjects
./.venv/bin/python scripts/behavioral/calculate_discount_rates.py --subjects sub-s061 sub-s130
```
-   **Input:** Raw `_events.tsv` files from the `onsets_dir`.
-   **Output:** Saves a `_discounting_with_sv.tsv` file for each subject in `derivatives/behavioral/sub-<ID>/` and an aggregated `discounting_model_fits.tsv` file in `derivatives/behavioral/`.

### Step 2: LSS Single-Trial Modeling

This step fits a first-level GLM for each trial to generate a beta map representing the brain's response to that specific trial.

**Command:**
```bash
./.venv/bin/python scripts/modeling/run_lss_modeling.py --subject sub-s061
```
-   **Input:** Preprocessed fMRI data from `fmriprep_dir` and the `_discounting_with_sv.tsv` file from Step 1.
-   **Output:** Saves a `_lss_beta_maps.nii.gz` file for the subject in `derivatives/lss_betas/sub-<ID>/`.

### Step 3: MVPA / Decoding Analysis

This step uses a machine learning classifier to test whether trial-by-trial behavioral variables can be predicted from the corresponding single-trial brain activity patterns. This is a direct test of whether a neural representation contains information about a specific variable.

The script uses a Support Vector Machine (`SVC` for classification, `SVR` for regression) with a cross-validation procedure (`StratifiedKFold` for classification to handle imbalanced data) to ensure the results are robust and generalizable.

You can specify any of the following as a target variable to decode: `choice`, `delay_to_reward`, `SVchosen`, `SVunchosen`, `SVsum`, and `SVdiff`.

**Command:**
```bash
# Example: Decode the 'choice' variable using the whole-brain mask
./.venv/bin/python scripts/mvpa/run_decoding_analysis.py --subject sub-s061 --target choice

# Example: Decode 'SVdiff' within a specific ROI
./.venv/bin/python scripts/mvpa/run_decoding_analysis.py --subject sub-s061 --target SVdiff --roi-path Masks/OFCmed_L_mask.nii.gz
```
-   **Input:** The `_lss_beta_maps.nii.gz` file (Step 2) and the behavioral data (Step 1). An optional ROI mask can also be provided.
-   **Output:** Saves a `_decoding-scores.tsv` file in `derivatives/mvpa/sub-<ID>/`. The filename will include the target variable and the ROI used (e.g., `..._target-choice_roi-vmPFC_decoding-scores.tsv`). The 'scores' column in this file contains the accuracy (for classification) or R^2 value (for regression) for each fold of the cross-validation.

### Step 4: Representational Similarity Analysis (RSA)

This step is the core of the representational geometry analysis. It tests the hypothesis that the structure of neural patterns is correlated with the structure of key behavioral variables.

The script automatically performs the following actions:
1.  Creates a **Neural RDM** from the trial-by-trial beta maps, which represents the measured geometry of brain activity.
2.  Creates multiple **Theoretical RDMs**, one for each of the following behavioral variables: `choice`, `delay_to_reward`, `SVchosen`, `SVunchosen`, `SVsum`, and `SVdiff`. Each of these matrices models a hypothetical geometry based on that variable.
3.  Calculates the Spearman's rank correlation between the Neural RDM and each of the Theoretical RDMs.

This provides a direct measure of how well each behavioral variable accounts for the representational geometry in the brain.

**Command:**
```bash
# Example: Run whole-brain RSA
./.venv/bin/python scripts/rsa/run_rsa_analysis.py --subject sub-s061 --analysis-type whole_brain

# Example: Run RSA within a specific ROI
./.venv/bin/python scripts/rsa/run_rsa_analysis.py --subject sub-s061 --analysis-type roi --roi-path Masks/OFCmed_L_mask.nii.gz

# Example: Run searchlight RSA
./.venv/bin/python scripts/rsa/run_rsa_analysis.py --subject sub-s061 --analysis-type searchlight
```
-   **Input:** The `_lss_beta_maps.nii.gz` file (Step 2) and the behavioral data (Step 1).
-   **Output:** Saves several files in `derivatives/rsa/sub-<ID>/`:
    -   `_rsa-results.tsv`: A table containing the Spearman's correlation coefficient (`rho`) for each behavioral variable. This is the main result file.
    -   `rdms/`: A directory containing the calculated Neural and Theoretical RDMs, which can be used for visualization.
    -   `searchlight_maps/`: If running a searchlight analysis, this directory will contain Nifti images where the value of each voxel is the correlation score for the searchlight centered at that voxel.

### Step 5: Visualization

This step generates 2D embeddings of the neural patterns to visualize the representational geometry using t-SNE.

**Command:**
```bash
# Example: Generate a t-SNE embedding for a subject, colored by 'SVchosen'
./.venv/bin/python scripts/visualization/plot_embeddings.py --subject sub-s061 --color-by SVchosen

# Example: Run the same analysis but within a specific ROI
./.venv/bin/python scripts/visualization/plot_embeddings.py --subject sub-s061 --color-by SVchosen --roi-path Masks/OFCmed_L_mask.nii.gz
```
-   **Input:** The `_lss_beta_maps.nii.gz` file (Step 2) and the behavioral data (Step 1).
-   **Output:** Saves a `.png` plot in `derivatives/visualization/sub-<ID>/`.

### Step 6: Automating Subject-Level Analyses

To improve efficiency, the `scripts/` directory contains wrapper scripts to run a full suite of analyses for a single subject with a single command.

**Commands:**
```bash
# Run all MVPA decoding analyses for a subject across all target variables and ROIs
./scripts/run_all_decoding_for_subject.sh sub-s061

# Run all t-SNE visualizations for a subject across all target variables and ROIs
./scripts/visualization/run_all_visualizations_for_subject.sh sub-s061
```

These scripts are a convenient way to process a single subject completely before scaling up to the full dataset on the HPC.

## 7. HPC Execution with SLURM

This pipeline is designed to be scaled up to run on a high-performance computing (HPC) cluster using the SLURM workload manager. The `slurm/` directory contains wrapper scripts (`submit_all_*.sh`) that automate the process of submitting jobs for all subjects found in your BIDS directory.

### Prerequisites

1.  Ensure this project repository is cloned on the HPC.
2.  Set up the Python environment using `uv` on the HPC, following the same steps as in the "Setup and Installation" section.
3.  Make sure you have correctly filled out the `hpc` section of your `config/project_config.yaml` file with the absolute paths to your data on the cluster.

### Submitting Jobs

To submit jobs, you run the corresponding `submit_all_*.sh` script. These scripts will find all `sub-*` directories in your BIDS directory and submit an `sbatch` job for each one.

**Important:** Before running, make sure the wrapper scripts are executable:
```bash
chmod +x slurm/submit_all_behavioral.sh
chmod +x slurm/submit_all_modeling.sh
chmod +x slurm/submit_all_decoding.sh
chmod +x slurm/submit_all_rsa.sh
```

**Example Commands:**

```bash
# Submit behavioral analysis for all subjects
./slurm/submit_all_behavioral.sh

# Submit LSS modeling for all subjects
./slurm/submit_all_modeling.sh

# Submit decoding analysis for all subjects and all target variables
./slurm/submit_all_decoding.sh

# Submit whole-brain RSA for all subjects
./slurm/submit_all_rsa.sh --analysis-type whole_brain

# Submit searchlight RSA for all subjects
./slurm/submit_all_rsa.sh --analysis-type searchlight
```

The scripts accept command-line arguments to specify the configuration file and environment, which default to the correct values (`--config config/project_config.yaml` and `--env hpc`). The SLURM output and error logs will be saved to the `slurm/` directory.
