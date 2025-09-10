import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import gamma
from nilearn.glm.first_level import FirstLevelModel
from scripts.utils import load_concatenated_subject_data, load_config, setup_logging
from typing import Dict, Any
import logging
from nilearn import image
from functools import reduce

def run_standard_glm_for_subject(subject_data: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Runs a standard GLM with parametric modulators across one or more runs,
    leveraging Nilearn's automatic design matrix creation.
    """
    subject_id = subject_data['subject_id']
    bold_imgs = subject_data['bold_imgs']
    mask_file = subject_data['mask_file']
    events_df = subject_data['events_df']
    confounds_dfs = subject_data['confounds_dfs']
    derivatives_dir = subject_data['derivatives_dir']

    logging.info(f"--- Running Standard GLM for {subject_id} on {len(bold_imgs)} run(s) ---")
    output_dir = derivatives_dir / 'standard_glm' / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Pre-computation Data Cleaning ---
    modulator_cols = [m for m in params['glm']['contrasts'] if m in events_df.columns and m != 'decision']
    if events_df[modulator_cols].isnull().values.any():
        logging.warning(f"NaNs found in modulator columns for {subject_id}. Filling with 0.")
        events_df[modulator_cols] = events_df[modulator_cols].fillna(0)

    if 'choice' in events_df.columns:
        events_df['choice'] = (events_df['choice'] == 'larger_later').astype(int)

    # --- Data Loading ---
    bold_imgs, confounds_dfs, events_df = subject_data['bold'], subject_data['confounds'], subject_data['events']
    
    # CRITICAL FIX: Standardize confound columns across all runs for this subject
    if confounds_dfs:
        common_confounds = list(reduce(set.intersection, [set(df.columns) for df in confounds_dfs]))
        confounds_dfs = [df[common_confounds] for df in confounds_dfs]
        logging.info(f"Standardized confounds to {len(common_confounds)} common columns across {len(confounds_dfs)} runs.")

    # --- Model Specification ---
    glm = FirstLevelModel(
        t_r=params['t_r'],
        slice_time_ref=params['slice_time_ref'],
        hrf_model=params['glm']['hrf_model'],
        drift_model=params['glm']['drift_model'],
        mask_img=mask_file,
        signal_scaling=False,
        smoothing_fwhm=params['smoothing_fwhm'],
        minimize_memory=False # Keep design matrices for inspection
    )

    # --- Prepare Events for Each Run ---
    events_per_run = []
    cleaned_confounds_dfs = []
    final_bold_imgs = []

    # Define the full set of potential modulator columns based on the config
    all_modulator_cols = params['analysis_params']['glm']['parametric_modulators']

    for i, (bold_img, confounds_df) in enumerate(zip(bold_imgs, confounds_dfs)):
        run_number = i + 1
        logging.info(f"  - Preparing data for run {run_number}/{len(bold_imgs)}")
        
        run_events_df = events_df[events_df['run'] == run_number].copy()

        if run_events_df.empty:
            logging.warning(f"No events found for run {run_number}. Excluding this run from the GLM.")
            continue

        final_bold_imgs.append(bold_img)
        cleaned_confounds_dfs.append(confounds_df)

        # Normalize onsets to be relative to the start of the run's events
        first_onset_in_run = run_events_df['onset'].min()
        run_events_df['onset'] -= first_onset_in_run
        
        run_events_df['trial_type'] = 'mean'
        
        # --- Parametric Modulator Processing ---
        # 1. Ensure all potential modulator columns exist, filling with 0 if absent
        for col in all_modulator_cols:
            if col not in run_events_df.columns:
                run_events_df[col] = 0
        
        # 2. Mean-center any non-zero modulators for this run
        for col in all_modulator_cols:
            if run_events_df[col].std() > 0:
                run_events_df[col] -= run_events_df[col].mean()

        # 3. Identify and remove any zero-variance modulators for this run
        final_modulators = [col for col in all_modulator_cols if run_events_df[col].std() > 0]
        if len(final_modulators) < len(all_modulator_cols):
            omitted = set(all_modulator_cols) - set(final_modulators)
            logging.info(f"    Omitting zero-variance modulators for this run: {list(omitted)}")

        nilearn_events_cols = ['onset', 'duration', 'trial_type'] + final_modulators
        events_per_run.append(run_events_df[nilearn_events_cols])
            
    # --- Fit the GLM with all valid runs ---
    if not final_bold_imgs:
        logging.error(f"No runs with valid event data found for {subject_id}. Aborting GLM.")
        return

    logging.info(f"Fitting GLM to {len(final_bold_imgs)} valid run(s)...")
    glm.fit(final_bold_imgs, events=events_per_run, confounds=cleaned_confounds_dfs)

    # --- Define and Compute Contrasts ---
    design_matrices = glm.design_matrices_
    
    # In case of multiple runs, we concatenate. For single run, this is just the one matrix.
    full_design_matrix = pd.concat(design_matrices, ignore_index=True)
    
    contrasts_to_compute = ['mean'] + params['analysis_params']['glm']['parametric_modulators']
    
    for contrast_name in contrasts_to_compute:
        # Identify all columns in the design matrix related to this contrast
        if contrast_name == 'mean':
            contrast_cols = [col for col in full_design_matrix.columns if col.startswith('mean') and 'x' not in col]
        else:
            contrast_cols = [col for col in full_design_matrix.columns if col.startswith(f'meanx{contrast_name}')]

        if not contrast_cols:
            logging.warning(f"Contrast '{contrast_name}' not found in any design matrix (likely omitted due to zero variance). Skipping.")
            continue

        # Build the contrast vector, correctly averaging over all runs
        contrast_vector = np.zeros(full_design_matrix.shape[1])
        weight = 1.0 / len(contrast_cols)
        for col in contrast_cols:
            contrast_vector[full_design_matrix.columns.get_loc(col)] = weight
        
        logging.info(f"  - Computing contrast for '{contrast_name}' (averaging over {len(contrast_cols)} column(s))...")
        z_map = glm.compute_contrast(contrast_vector, output_type='z_score')
        
        contrast_filename = output_dir / f"contrast-{contrast_name}_zmap.nii.gz"
        z_map.to_filename(contrast_filename)
        logging.info(f"Saved z-map to: {contrast_filename.resolve()}")

def make_contrast_vector(columns, condition_weights: Dict[str, float]) -> np.ndarray:
    """Helper function to create a contrast vector from a dictionary of weights."""
    vector = np.zeros(len(columns))
    for condition, weight in condition_weights.items():
        if condition in columns:
            vector[columns.get_loc(condition)] = weight
        else:
            logging.warning(f"Condition '{condition}' not found in design matrix columns. Skipping in contrast.")
    return vector

def main() -> None:
    parser = argparse.ArgumentParser(description="Run standard GLM for a single subject.")
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to the project config file')
    parser.add_argument('--env', type=str, required=True, choices=['local', 'hpc'], help='Environment to run on')
    parser.add_argument('--subject', type=str, required=True, help='The subject ID to process')
    args = parser.parse_args()

    setup_logging()

    # --- Load Data & Config ---
    subject_data = load_concatenated_subject_data(args.config, args.env, args.subject)
    config = load_config(args.config)
    analysis_params = config['analysis_params']
    
    # --- Run Analysis ---
    run_standard_glm_for_subject(subject_data, analysis_params)


if __name__ == "__main__":
    main()
