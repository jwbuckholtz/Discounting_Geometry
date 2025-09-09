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

    # --- GLM Setup ---
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
    # We will create a list of event DataFrames, one for each run.
    events_per_run = []
    cleaned_confounds_dfs = []
    final_bold_imgs = []

    # Define the full set of potential modulator columns based on the config
    all_modulator_cols = [m for m in params['glm']['contrasts'] if m != 'decision']

    for i, (bold_img, confounds_df) in enumerate(zip(bold_imgs, confounds_dfs)):
        run_number = i + 1
        logging.info(f"  - Preparing data for run {run_number}/{len(bold_imgs)}")
        
        # Isolate events for the current run
        run_events_df = events_df[events_df['run'] == run_number].copy()

        # If a run has no events, we must exclude it completely from the analysis
        if run_events_df.empty:
            logging.warning(f"No events found for run {run_number}. Excluding this run from the GLM.")
            continue

        # This run is valid, so we keep its data
        final_bold_imgs.append(bold_img)

        # Clean confounds for this run
        if confounds_df.isnull().values.any():
            logging.warning(f"NaNs found in confounds for run {run_number}. Filling with 0.")
            cleaned_confounds_dfs.append(confounds_df.fillna(0))
        else:
            cleaned_confounds_dfs.append(confounds_df)

        # Normalize onsets to be relative to the start of the run
        first_onset_in_run = run_events_df['onset'].min()
        run_events_df['onset'] -= first_onset_in_run
        logging.info(f"Normalizing onsets for run {run_number} by subtracting {first_onset_in_run:.4f}s")

        # Add a 'trial_type' column for Nilearn's GLM
        run_events_df['trial_type'] = 'decision'
        
        # Ensure all potential modulator columns exist for this run, filling with 0 if absent
        for col in all_modulator_cols:
            if col not in run_events_df.columns:
                run_events_df[col] = 0
        
        # Select and order columns consistently
        nilearn_events_cols = ['onset', 'duration', 'trial_type'] + all_modulator_cols
        events_per_run.append(run_events_df[nilearn_events_cols])
            
    # --- Fit the GLM with all valid runs ---
    if not final_bold_imgs:
        logging.error(f"No runs with valid event data found for {subject_id}. Aborting GLM.")
        return

    logging.info(f"Fitting GLM to {len(final_bold_imgs)} valid run(s)...")
    glm.fit(final_bold_imgs, events=events_per_run, confounds=cleaned_confounds_dfs)

    # --- Define and Compute Contrasts ---
    # Inspect the design matrix to get the order of regressors
    final_design_matrix_columns = glm.design_matrices_[-1].columns.tolist()
    
    # Create a dict for all possible contrasts based on the config
    contrasts = {}
    for contrast_id in params['glm']['contrasts']:
        # Create a contrast vector: an array of zeros with a 1 at the position of the desired regressor
        contrast_vector = np.zeros(len(final_design_matrix_columns))
        
        # Find the column index for the current contrast
        try:
            # The unmodulated event regressor is named after the trial_type
            regressor_name = 'decision' if contrast_id == 'decision' else contrast_id
            regressor_index = final_design_matrix_columns.index(regressor_name)
            contrast_vector[regressor_index] = 1
            contrasts[contrast_id] = contrast_vector
        except ValueError:
            logging.warning(f"Could not find regressor '{contrast_id}' in the design matrix. Skipping contrast.")

    for contrast_id, contrast_vector in contrasts.items():
        try:
            logging.info(f"  - Computing contrast for '{contrast_id}'")
            z_map = glm.compute_contrast(contrast_vector, output_type='z_score')
            output_filename = output_dir / 'z_maps' / f"{contrast_id}_zmap.nii.gz"
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            z_map.to_filename(output_filename)
            logging.info(f"Saved z-map to: {output_filename.resolve()}")

            if output_filename.exists():
                logging.info(f"  [SUCCESS] File check passed for {output_filename.name}")
            else:
                logging.error(f"  [FAILURE] File check failed for {output_filename.name}")

        except Exception as e:
            logging.error(f"Contrast '{contrast_id}' could not be computed. Error: {e}. Skipping.")


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
