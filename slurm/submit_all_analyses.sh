#!/bin/bash
# ==============================================================================
# Master SLURM Job Submission Script for the DecodingDD Project
# ==============================================================================
# This script is the main entry point for running the entire analysis pipeline
# on the HPC. It submits jobs for each major analysis step in the correct order.
#
# Usage:
# ./slurm/submit_all_analyses.sh
#
# Steps:
# 1. Submits behavioral data processing for all subjects.
# 2. Waits for behavioral jobs to finish.
# 3. Submits LSS and Standard GLM modeling jobs for all subjects.
# 4. Waits for modeling jobs to finish.
# 5. Submits RSA and Decoding jobs for all subjects.
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting full analysis pipeline submission..."

# --- Step 1: Behavioral Analysis ---
echo "Submitting behavioral analysis jobs..."
BEHAVIORAL_JOB_ID=$(./slurm/submit_behavioral_analysis.sbatch | awk '{print $4}')
echo "Behavioral analysis submitted with Job ID: $BEHAVIORAL_JOB_ID"

# --- Step 2: First-Level Modeling (Depends on Behavioral) ---
echo "Submitting first-level modeling jobs (will run after behavioral)..."
MODELING_JOB_ID=$(sbatch --dependency=afterok:$BEHAVIORAL_JOB_ID slurm/submit_all_modeling.sh | awk '{print $4}')
echo "Modeling jobs submitted with Job ID: $MODELING_JOB_ID"

# --- Step 3: MVPA & RSA (Depends on Modeling) ---
echo "Submitting RSA jobs (will run after modeling)..."
RSA_JOB_ID=$(sbatch --dependency=afterok:$MODELING_JOB_ID slurm/submit_all_rsa.sh | awk '{print $4}')
echo "RSA jobs submitted with Job ID: $RSA_JOB_ID"

echo "Submitting Decoding jobs (will run after modeling)..."
DECODING_JOB_ID=$(sbatch --dependency=afterok:$MODELING_JOB_ID slurm/submit_all_decoding.sh | awk '{print $4}')
echo "Decoding jobs submitted with Job ID: $DECODING_JOB_ID"

echo "All jobs submitted successfully."
echo "Monitor job status with 'squeue -u \$USER'"
