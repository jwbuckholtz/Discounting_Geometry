import argparse
from pathlib import Path
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image
from scripts.utils import load_concatenated_subject_data, load_config, setup_logging
from typing import Dict, Any
import logging

def run_lss_for_subject(subject_data: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Runs the LSS modeling to estimate single-trial beta maps from one or more runs.
    """
    subject_id = subject_data['subject_id']
    bold_imgs = subject_data['bold_imgs']
    mask_file = subject_data['mask_file']
    events_df = subject_data['events_df']
    confounds_dfs = subject_data['confounds_dfs']
    derivatives_dir = subject_data['derivatives_dir']

    logging.info(f"--- Running LSS Model for {subject_id} on {len(bold_imgs)} run(s) ---")

    # --- CRITICAL FIX: Normalize onsets to be relative to the start of each run ---
    # The onset times in the behavioral files are cumulative across the session.
    # We must create a new events dataframe where onsets are relative to their run's start time.
    corrected_events_list = []
    for run_number in sorted(events_df['run'].unique()):
        run_events_df = events_df[events_df['run'] == run_number].copy()
        if not run_events_df.empty:
            first_onset_in_run = run_events_df['onset'].min()
            run_events_df['onset'] -= first_onset_in_run
            logging.info(f"Normalizing onsets for run {run_number} by subtracting {first_onset_in_run:.4f}s")
            corrected_events_list.append(run_events_df)
    
    # Overwrite the original events_df with the corrected one
    events_df = pd.concat(corrected_events_list)

    # --- Pre-computation Data Cleaning ---
    # Fill any NaNs from the confounds files to prevent crashes.
    # Use a list comprehension to create a new list of cleaned dataframes.
    cleaned_confounds_dfs = []
    for conf_df in confounds_dfs:
        if conf_df.isnull().values.any():
            logging.warning(f"NaNs found in confounds for {subject_id}. Filling with 0.")
            cleaned_confounds_dfs.append(conf_df.fillna(0))
        else:
            cleaned_confounds_dfs.append(conf_df)

    # Define the GLM using parameters from the config file
    glm = FirstLevelModel(
        t_r=params['t_r'],
        slice_time_ref=params['slice_time_ref'],
        hrf_model='glover',
        drift_model='cosine',
        mask_img=mask_file,
        signal_scaling=False,
    )

    # --- LSS Modeling: Iterate Through Each Trial ---
    beta_maps = []
    # Get a list of all trial indices from the main events dataframe
    trial_indices = events_df.index.tolist()

    for trial_idx in trial_indices:
        # Isolate the run number for the current trial
        trial_run = events_df.loc[trial_idx, 'run']
        
        # Create the LSS events dataframe for this specific trial
        # All other trials (even in other runs) are modeled as a single 'other' regressor
        lss_events_df = events_df[['onset', 'duration', 'run']].copy()
        lss_events_df['trial_type'] = 'other'
        lss_events_df.loc[trial_idx, 'trial_type'] = f'trial_{trial_idx}'

        # --- Run-Specific Fitting for LSS ---
        # Although nilearn's LSS implementation can handle multi-run data implicitly
        # by matching onsets to the correct run's timeline, we still need to provide
        # the BOLD images and confounds as a list.
        glm.fit(bold_imgs, events=lss_events_df, confounds=cleaned_confounds_dfs)
        
        # Compute the contrast for the single target trial
        beta_map = glm.compute_contrast(f'trial_{trial_idx}', output_type='effect_size')
        beta_maps.append(beta_map)

        if (trial_idx + 1) % 10 == 0:
            logging.info(f"  - Completed LSS for trial {trial_idx + 1}/{len(trial_indices)}")

    # Concatenate all beta maps into a single 4D NIfTI image
    beta_maps_img = image.concat_imgs(beta_maps)

    # Save the beta maps NIfTI image
    output_dir = derivatives_dir / 'lss_betas' / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{subject_id}_lss_beta_maps.nii.gz"
    beta_maps_img.to_filename(output_filename)
    logging.info(f"LSS beta maps saved to {output_filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LSS modeling for a single subject.")
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to the project config file')
    parser.add_argument('--env', type=str, required=True, choices=['local', 'hpc'], help='Environment to run on')
    parser.add_argument('--subject', type=str, required=True, help='The subject ID to process')
    args = parser.parse_args()

    setup_logging()

    # --- Load Data & Config ---
    # This function now loads and prepares data from all runs for the subject
    subject_data = load_concatenated_subject_data(args.config, args.env, args.subject)
    config = load_config(args.config)
    analysis_params = config['analysis_params']
    
    # --- Run Analysis ---
    run_lss_for_subject(subject_data, analysis_params)


if __name__ == "__main__":
    main()
