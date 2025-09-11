# SLURM Script Hierarchy and Dependencies

## Visual Overview

```
📁 slurm/
├── 🚀 prepare_slurm_environment.sh    ⭐ START HERE (setup)
├── 
├── 📊 MAIN ANALYSIS WORKFLOWS:
├── └── submit_behavioral_analysis.sbatch      → Step 1: Behavioral
├── └── submit_glm_array_dynamic.sh           → Step 2: GLM (all subjects)
├── └── submit_all_group_glms.sh              → Step 3: Group analysis
├── 
├── 🔧 DYNAMIC SUBMISSION SCRIPTS:
├── ├── submit_glm_array_dynamic.sh           (calls submit_glm_batch.sbatch)
├── ├── submit_lss_array_dynamic.sh           (calls submit_lss_batch.sbatch)
├── └── get_subject_count.sh                  (helper for dynamic scripts)
├── 
├── ⚙️ BATCH WORKER SCRIPTS:
├── ├── submit_glm_batch.sbatch               (individual GLM jobs)
├── ├── submit_lss_batch.sbatch               (individual LSS jobs)
├── └── submit_group_glm.sbatch               (individual group jobs)
├── 
├── 🎯 SINGLE-JOB SCRIPTS:
├── ├── submit_standard_glm.sbatch            (single subject GLM)
├── ├── submit_group_glm.sbatch               (single contrast group)
├── └── submit_visualization.sbatch           (single subject viz)
├── 
└── 📋 TEMPLATES (advanced users):
    ├── submit_decoding_template.sbatch       (MVPA template)
    └── submit_rsa_template.sbatch            (RSA template)
```

## Script Categories and Usage Patterns

### 🚀 ESSENTIAL SCRIPTS (Use These)

#### 1. Environment Setup
```bash
./slurm/prepare_slurm_environment.sh
```
- **Always run first**
- Creates directories, validates environment
- One-time setup per session

#### 2. Main Analysis Pipeline
```bash
# Full pipeline in order:
sbatch slurm/submit_behavioral_analysis.sbatch      # Step 1
./slurm/submit_glm_array_dynamic.sh                 # Step 2  
./slurm/submit_all_group_glms.sh                    # Step 3
```

### 🔧 DYNAMIC SCRIPTS (Recommended)

These automatically calculate array bounds and handle environment setup:

```bash
./slurm/submit_glm_array_dynamic.sh    # GLM for all subjects
./slurm/submit_lss_array_dynamic.sh    # LSS for all subjects
./slurm/submit_all_group_glms.sh       # Group analysis for all contrasts
```

**Why use these?**
- Automatically count subjects
- No hard-coded array bounds
- Built-in validation
- Create required directories

### ⚙️ BATCH WORKER SCRIPTS (Internal Use)

These are called by dynamic scripts - you don't submit them directly:

```bash
# DON'T RUN THESE DIRECTLY:
submit_glm_batch.sbatch     # Called by submit_glm_array_dynamic.sh
submit_lss_batch.sbatch     # Called by submit_lss_array_dynamic.sh
```

### 🎯 SINGLE-JOB SCRIPTS (Testing/Specific Use)

For testing or rerunning specific components:

```bash
# Single subject GLM (testing)
export SUBJECT_ID=sub-s061
sbatch slurm/submit_standard_glm.sbatch

# Single contrast group analysis (rerun)
export CONTRAST=choice
sbatch slurm/submit_group_glm.sbatch

# Single subject visualization
export SUBJECT_ID=sub-s061 ROI_DIR=Masks/
sbatch slurm/submit_visualization.sbatch
```

### 📋 TEMPLATES (Advanced/Custom Analysis)

For specialized analyses - customize before use:

```bash
# MVPA decoding (customize for your targets)
export TARGET=choice
sbatch slurm/templates/submit_decoding_template.sbatch

# RSA analysis (customize for your analysis type)
export ANALYSIS_TYPE=behavioral
sbatch slurm/templates/submit_rsa_template.sbatch
```

## Dependency Chain

```
prepare_slurm_environment.sh
         ↓
submit_behavioral_analysis.sbatch
         ↓
submit_glm_array_dynamic.sh
    ├─→ get_subject_count.sh
    └─→ submit_glm_batch.sbatch (array job)
         ↓
submit_all_group_glms.sh
    └─→ submit_group_glm.sbatch (multiple jobs)
         ↓
Optional:
├─→ submit_lss_array_dynamic.sh
│   ├─→ get_subject_count.sh
│   └─→ submit_lss_batch.sbatch (array job)
└─→ submit_visualization.sbatch
```

## Decision Tree: Which Script to Use?

```
🤔 What do you want to do?

├─ 🚀 First time setup?
│  └─ ./slurm/prepare_slurm_environment.sh

├─ 📊 Run complete analysis pipeline?
│  ├─ sbatch slurm/submit_behavioral_analysis.sbatch
│  ├─ ./slurm/submit_glm_array_dynamic.sh
│  └─ ./slurm/submit_all_group_glms.sh

├─ 🔬 Test with single subject?
│  └─ sbatch slurm/submit_standard_glm.sbatch

├─ 🎯 Rerun specific contrast?
│  └─ sbatch slurm/submit_group_glm.sbatch

├─ 📈 Generate visualizations?
│  └─ sbatch slurm/submit_visualization.sbatch

├─ 🧠 Advanced MVPA analysis?
│  └─ sbatch slurm/templates/submit_decoding_template.sbatch

└─ 🔍 Advanced RSA analysis?
   └─ sbatch slurm/templates/submit_rsa_template.sbatch
```

## Environment Variable Requirements

### Universal (All Scripts)
```bash
export PROJECT_ROOT=/path/to/DecodingDD
```

### Array Jobs (Dynamic Scripts)
```bash
export PROJECT_ROOT=/path/to/DecodingDD
export BEHAVIORAL_DIR=/path/to/behavioral/data
```

### Single Jobs
```bash
export PROJECT_ROOT=/path/to/DecodingDD
export SUBJECT_ID=sub-s061
export CONFIG_FILE=config/project_config.yaml
export ENV=hpc
```

### Specialized Analysis
```bash
# Group analysis
export CONTRAST=choice

# Visualization
export ROI_DIR=Masks/

# MVPA
export TARGET=choice

# RSA
export ANALYSIS_TYPE=behavioral
```

## Common Pitfalls to Avoid

❌ **DON'T DO THIS:**
```bash
# Hard-coded array bounds (will break)
sbatch --array=0-102 slurm/submit_glm_batch.sbatch

# Running from wrong directory
cd /some/other/directory
./slurm/submit_glm_array_dynamic.sh

# Missing environment variables
./slurm/submit_glm_array_dynamic.sh  # No PROJECT_ROOT set
```

✅ **DO THIS INSTEAD:**
```bash
# Use dynamic scripts
export PROJECT_ROOT=/path/to/project
export BEHAVIORAL_DIR=/path/to/behavioral/data
./slurm/submit_glm_array_dynamic.sh

# Always set environment first
export PROJECT_ROOT=/path/to/DecodingDD
./slurm/prepare_slurm_environment.sh
```

## Quick Reference Card

### Standard Workflow
```bash
# 1. Setup
export PROJECT_ROOT=/path/to/DecodingDD
export BEHAVIORAL_DIR=/path/to/behavioral/data
./slurm/prepare_slurm_environment.sh

# 2. Analysis
sbatch slurm/submit_behavioral_analysis.sbatch
./slurm/submit_glm_array_dynamic.sh
./slurm/submit_all_group_glms.sh

# 3. Check progress
squeue -u $USER
ls logs/
```

This hierarchy guide ensures you always know exactly which script to use for your specific needs!
