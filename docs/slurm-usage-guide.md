# SLURM Scripts Usage Guide

## Overview
The DecodingDD project has a comprehensive SLURM infrastructure for running fMRI analyses on HPC clusters. This guide explains which scripts to use, when, and in what order.

## Quick Start: Typical Analysis Workflow

### 1. Environment Setup (One-time)
```bash
# Set required environment variables
export PROJECT_ROOT=/path/to/DecodingDD
export BEHAVIORAL_DIR=/path/to/behavioral/data

# Prepare environment and validate setup
./slurm/prepare_slurm_environment.sh
```

### 2. Complete Analysis Pipeline (Recommended Order)
```bash
# Step 1: Behavioral Analysis (single job)
sbatch slurm/submit_behavioral_analysis.sbatch

# Step 2: First-level GLM Analysis (array job, all subjects)
./slurm/submit_glm_array_dynamic.sh

# Step 3: LSS Analysis (array job, all subjects) - Optional
./slurm/submit_lss_array_dynamic.sh

# Step 4: Group-level GLM Analysis (multiple contrasts)
./slurm/submit_all_group_glms.sh

# Step 5: Visualization (single subject example)
export SUBJECT_ID=sub-s061
export ROI_DIR=Masks/
sbatch slurm/submit_visualization.sbatch
```

## Detailed Script Reference

### Environment and Setup Scripts

#### `slurm/prepare_slurm_environment.sh` ⭐ START HERE
**Purpose**: One-time environment setup and validation
**When to use**: Before any SLURM job submission
**Usage**:
```bash
./slurm/prepare_slurm_environment.sh
```
**What it does**:
- Creates all required directories (logs/, derivatives/, etc.)
- Validates PROJECT_ROOT and environment
- Checks for Python, SLURM, virtual environment
- Provides usage guidance

#### `slurm/get_subject_count.sh`
**Purpose**: Helper script to count subjects for array jobs
**When to use**: Called automatically by dynamic submission scripts
**Usage**:
```bash
# Used internally, but you can run manually:
export BEHAVIORAL_DIR=/path/to/behavioral/data
./slurm/get_subject_count.sh
```

### Core Analysis Scripts

#### `slurm/submit_glm_array_dynamic.sh` ⭐ MAIN GLM ANALYSIS
**Purpose**: Submit first-level GLM analysis for all subjects
**When to use**: Primary analysis step, after behavioral analysis
**Usage**:
```bash
# Set environment variables first
export PROJECT_ROOT=/path/to/project
export BEHAVIORAL_DIR=/path/to/behavioral/data

# Submit dynamic array job
./slurm/submit_glm_array_dynamic.sh
```
**What it does**:
- Automatically calculates number of subjects
- Submits array job with correct bounds (0 to N-1)
- Runs separate GLM models for each regressor
- Creates model-specific output directories

#### `slurm/submit_lss_array_dynamic.sh`
**Purpose**: Submit LSS (Least Squares Single-trial) analysis for all subjects
**When to use**: After GLM analysis, for single-trial estimates
**Usage**:
```bash
export PROJECT_ROOT=/path/to/project
export BEHAVIORAL_DIR=/path/to/behavioral/data
./slurm/submit_lss_array_dynamic.sh
```

#### `slurm/submit_all_group_glms.sh` ⭐ GROUP ANALYSIS
**Purpose**: Submit group-level analysis for all contrasts
**When to use**: After first-level GLM is complete for all subjects
**Usage**:
```bash
export PROJECT_ROOT=/path/to/project
./slurm/submit_all_group_glms.sh
```
**What it does**:
- Submits separate jobs for each contrast (choice, SVchosen, etc.)
- Runs group-level statistics
- Creates group-level statistical maps

### Individual Job Scripts (Manual Submission)

#### `slurm/submit_behavioral_analysis.sbatch`
**Purpose**: Analyze behavioral data and compute discount rates
**When to use**: First step in analysis pipeline
**Usage**:
```bash
export PROJECT_ROOT=/path/to/project
sbatch slurm/submit_behavioral_analysis.sbatch
```

#### `slurm/submit_standard_glm.sbatch`
**Purpose**: Run GLM analysis for a single subject
**When to use**: Testing or rerunning specific subjects
**Usage**:
```bash
export PROJECT_ROOT=/path/to/project
export SUBJECT_ID=sub-s061
export CONFIG_FILE=config/project_config.yaml
export ENV=hpc
sbatch slurm/submit_standard_glm.sbatch
```

#### `slurm/submit_group_glm.sbatch`
**Purpose**: Run group-level analysis for a single contrast
**When to use**: Testing or rerunning specific contrasts
**Usage**:
```bash
export PROJECT_ROOT=/path/to/project
export CONFIG_FILE=config/project_config.yaml
export ENV=hpc
export CONTRAST=choice
sbatch slurm/submit_group_glm.sbatch
```

#### `slurm/submit_visualization.sbatch`
**Purpose**: Generate visualizations for a single subject
**When to use**: After analyses complete, for specific subjects
**Usage**:
```bash
export PROJECT_ROOT=/path/to/project
export SUBJECT_ID=sub-s061
export ROI_DIR=Masks/
sbatch slurm/submit_visualization.sbatch
```

### Low-level Batch Scripts (Used by Dynamic Scripts)

These are called automatically by the dynamic scripts above. You typically don't submit them directly:

- `slurm/submit_glm_batch.sbatch` - Individual GLM array job worker
- `slurm/submit_lss_batch.sbatch` - Individual LSS array job worker

### Template Scripts (For Advanced Users)

#### `slurm/templates/submit_decoding_template.sbatch`
**Purpose**: Template for MVPA decoding analysis
**When to use**: Advanced analysis, customize for specific decoding targets
**Usage**:
```bash
export PROJECT_ROOT=/path/to/project
export SUBJECT_ID=sub-s061
export TARGET=choice
export CONFIG_FILE=config/project_config.yaml
export ENV=hpc
sbatch slurm/templates/submit_decoding_template.sbatch
```

#### `slurm/templates/submit_rsa_template.sbatch`
**Purpose**: Template for Representational Similarity Analysis
**When to use**: Advanced analysis, customize for specific RSA types
**Usage**:
```bash
export PROJECT_ROOT=/path/to/project
export SUBJECT_ID=sub-s061
export ANALYSIS_TYPE=behavioral
export CONFIG_FILE=config/project_config.yaml
export ENV=hpc
sbatch slurm/templates/submit_rsa_template.sbatch
```

## Common Workflows

### Full Analysis Pipeline
```bash
# 1. Setup (once)
export PROJECT_ROOT=/path/to/DecodingDD
export BEHAVIORAL_DIR=/path/to/behavioral/data
./slurm/prepare_slurm_environment.sh

# 2. Run complete pipeline
sbatch slurm/submit_behavioral_analysis.sbatch
./slurm/submit_glm_array_dynamic.sh
./slurm/submit_all_group_glms.sh

# 3. Optional: LSS and visualization
./slurm/submit_lss_array_dynamic.sh
export SUBJECT_ID=sub-s061 ROI_DIR=Masks/
sbatch slurm/submit_visualization.sbatch
```

### Testing/Development Workflow
```bash
# Test with single subject
export PROJECT_ROOT=/path/to/project
export SUBJECT_ID=sub-s061
export CONFIG_FILE=config/project_config.yaml
export ENV=hpc

sbatch slurm/submit_standard_glm.sbatch
```

### Rerun Specific Components
```bash
# Rerun group analysis for specific contrast
export PROJECT_ROOT=/path/to/project
export CONTRAST=choice
export CONFIG_FILE=config/project_config.yaml
export ENV=hpc

sbatch slurm/submit_group_glm.sbatch
```

## Environment Variables Reference

### Required for All Scripts
- `PROJECT_ROOT`: Path to DecodingDD project directory

### Required for Array Jobs
- `BEHAVIORAL_DIR`: Path to directory containing subject behavioral data

### Required for Individual Jobs
- `SUBJECT_ID`: Subject identifier (e.g., "sub-s061")
- `CONFIG_FILE`: Path to config file (usually "config/project_config.yaml")
- `ENV`: Environment ("hpc" or "local")

### Analysis-Specific
- `CONTRAST`: Contrast name for group analysis
- `TARGET`: Decoding target for MVPA
- `ANALYSIS_TYPE`: RSA analysis type
- `ROI_DIR`: Directory containing ROI masks

## Troubleshooting

### Common Issues
1. **"PROJECT_ROOT not set"**: Export PROJECT_ROOT before running any script
2. **"No subjects found"**: Check BEHAVIORAL_DIR points to correct location
3. **"Array bounds out of range"**: Use dynamic scripts, not manual array submission
4. **"Logs directory missing"**: Run prepare_slurm_environment.sh first

### Check Job Status
```bash
# Check running jobs
squeue -u $USER

# Check specific job details
scontrol show job <JOBID>

# Check logs
ls logs/
tail logs/glm_batch_<JOBID>_<TASKID>.out
```

## Best Practices

1. **Always start with**: `./slurm/prepare_slurm_environment.sh`
2. **Use dynamic scripts**: Prefer `*_dynamic.sh` over manual array submission
3. **Set environment variables**: Required for all scripts to work properly
4. **Check logs**: Monitor logs/ directory for job progress and errors
5. **Run in order**: Behavioral → GLM → Group → Visualization
6. **Test first**: Use single-subject scripts for testing before array jobs

This guide ensures you can confidently navigate and use the sophisticated SLURM infrastructure!
