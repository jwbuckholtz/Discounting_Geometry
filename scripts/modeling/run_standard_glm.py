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

    for i, confounds_df in enumerate(confounds_dfs):
        run_number = i + 1
        logging.info(f"  - Preparing data for run {run_number}/{len(bold_imgs)}")
        
        # Clean confounds for this run
        if confounds_df.isnull().values.any():
            logging.warning(f"NaNs found in confounds for run {run_number}. Filling with 0.")
            cleaned_confounds_dfs.append(confounds_df.fillna(0))
        else:
            cleaned_confounds_dfs.append(confounds_df)

        # Isolate events for the current run
        run_events_df = events_df[events_df['run'] == run_number].copy()

        # Normalize onsets to be relative to the start of the run
        if not run_events_df.empty:
            first_onset_in_run = run_events_df['onset'].min()
            run_events_df['onset'] -= first_onset_in_run
            logging.info(f"Normalizing onsets for run {run_number} by subtracting {first_onset_in_run:.4f}s")
        else:
            logging.warning(f"No events found for run {run_number}. Skipping.")
            events_per_run.append(None) # Append None to keep lists aligned
            continue

        # Add a 'trial_type' column for Nilearn's GLM
        # All events are of the 'decision' type, and other columns become parametric modulators.
        run_events_df['trial_type'] = 'decision'
        
        # Select only the necessary columns for Nilearn
        modulator_names = [col for col in modulator_cols if col in run_events_df.columns]
        nilearn_events_cols = ['onset', 'duration', 'trial_type'] + modulator_names
        events_per_run.append(run_events_df[nilearn_events_cols])
            
    # --- Fit the GLM with all runs ---
    logging.info("Fitting GLM to all runs...")
    glm.fit(bold_imgs, events=events_per_run, confounds=cleaned_confounds_dfs)

    # --- Define and Compute Contrasts ---
    # Inspect the design matrix from the last run to get regressor names
    final_design_matrix = glm.design_matrices_[-1]
    
    # Create a dict for all possible contrasts based on the config
    contrasts = {}
    for contrast_id in params['glm']['contrasts']:
        if contrast_id == 'decision':
            # The unmodulated event regressor
            contrasts[contrast_id] = final_design_matrix['decision']
        elif contrast_id in final_design_matrix.columns:
            # Simple parametric modulators
            contrasts[contrast_id] = final_design_matrix[contrast_id]

    for contrast_id, contrast_formula in contrasts.items():
        try:
            logging.info(f"  - Computing contrast for '{contrast_id}'")
            z_map = glm.compute_contrast(contrast_formula, output_type='z_score')
            output_filename = output_dir / 'z_maps' / f"{contrast_id}_zmap.nii.gz"
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            z_map.to_filename(output_filename)
            logging.info(f"Saved z-map to: {output_filename.resolve()}")

            if output_filename.exists():
                logging.info(f"  [SUCCESS] File check passed for {output_filename.name}")
            else:
                logging.error(f"  [FAILURE] File check failed for {output_filename.name}")

        except ValueError:
            logging.warning(f"Contrast '{contrast_id}' could not be computed. It might be all zeros. Skipping.")


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
