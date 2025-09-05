# Decoding Delay Discounting (DecodingDD)

## 1. Project Overview

This project implements a comprehensive fMRI analysis pipeline to investigate the neural representations underlying delay discounting decisions. The primary goal is to leverage multivariate pattern analysis (MVPA) and representational similarity analysis (RSA) to decode and characterize the geometry of neural patterns related to subjective value and choice.

The analysis is designed to be modular, reproducible, and scalable, adhering to the "Better code, better science" principles. It includes stages for:
- Behavioral modeling to estimate discount rates.
- First-level fMRI modeling using both a standard GLM and Least-Squares-Separate (LSS).
- MVPA (decoding) to test for brain-based classification of choice.
- RSA to compare the representational geometry of neural patterns to theoretical models.

The entire pipeline is designed to be executed on both local machines for testing and on a high-performance computing (HPC) cluster using SLURM for large-scale analysis.

## 2. Repository Structure

```
/
├── config/               # Project and analysis parameter configuration
│   └── project_config.yaml
├── data/                 # Raw BIDS data (gitignored)
├── derivatives/          # All analysis outputs (gitignored)
├── scripts/              # All analysis and utility scripts
│   ├── behavioral/
│   ├── modeling/
│   ├── mvpa/
│   └── rsa/
├── slurm/                # SLURM job submission scripts
│   ├── templates/
│   └── submit_all_analyses.sh
├── tests/                # Automated test suite for the analysis scripts
├── .gitignore
└── pyproject.toml        # Project metadata and Python dependencies
```

## 3. Setup and Installation

This project uses `uv` for fast and reproducible Python environment management.

1.  **Clone the Repository**: `git clone <repository_url> && cd DecodingDD`
2.  **Create and Activate Environment**: `python3.9 -m venv .venv && source .venv/bin/activate`
3.  **Install Dependencies**: `pip install uv && uv pip sync pyproject.toml`

## 4. Configuration

All paths and analysis parameters are managed through a central configuration file: `config/project_config.yaml`. This allows you to adapt the pipeline to different computing environments and change analysis parameters without modifying the code.

Before running, you must edit this file to provide the correct absolute paths for your system(s) in the `local` and `hpc` sections. You can also modify analysis parameters (e.g., GLM contrasts, RSA models) in the `analysis_params` section.

## 5. Automated Testing with `pytest`

This project includes a comprehensive suite of automated tests to ensure the correctness and reliability of the core analysis functions. The tests are a critical component for ensuring scientific validity and have been instrumental in developing this pipeline.

The test suite provides coverage for all major analysis stages: `behavioral`, `modeling` (LSS and GLM), `mvpa`, and `rsa`.

-   **To run the tests**, first install the testing dependencies: `uv pip install -e ".[test]"`
-   Then, execute `pytest` from the project root: `pytest`

## 6. Running the Analysis Pipeline

The pipeline is designed to be run sequentially from the command line.

### Local Execution (for a single subject)

You can run individual scripts for testing or debugging on a local machine. All commands should be run from the project root.

1.  **Behavioral Analysis**: `python scripts/behavioral/calculate_discount_rates.py --env local --subjects sub-01`
2.  **LSS Modeling**: `python scripts/modeling/run_lss_model.py --env local --subject sub-01`
3.  **Standard GLM**: `python scripts/modeling/run_standard_glm.py --env local --subject sub-01`
4.  **RSA**: `python scripts/rsa/run_rsa_analysis.py --env local --subject sub-01 --analysis-type whole_brain`
5.  **MVPA**: `python scripts/mvpa/run_decoding_analysis.py sub-01 /path/to/bids /path/to/derivatives /path/to/mask.nii.gz`

### HPC Execution (for all subjects)

The entire pipeline can be submitted to a SLURM-managed HPC cluster with a single command. The system is designed to be robust, using job dependencies to ensure that each stage of the analysis runs only after its prerequisites have successfully completed.

1.  **Ensure `project_config.yaml` is correct**: The `hpc` section must contain the correct paths for your cluster environment.
2.  **Ensure submission scripts are executable**: `chmod +x slurm/*.sh`
3.  **Submit the master script**:
    ```bash
    ./slurm/submit_all_analyses.sh
    ```

This single command will:
1.  Submit the behavioral analysis job for all subjects.
2.  Queue the modeling jobs, which will automatically start after the behavioral job completes successfully.
3.  Queue the RSA and MVPA jobs, which will automatically start after the modeling jobs complete successfully.

You can monitor the progress of your jobs using `squeue -u $USER`. Logs for each job will be saved to the `slurm/logs/` directory.
