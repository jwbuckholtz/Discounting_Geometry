import argparse
from pathlib import Path
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image
from scripts.utils import load_config, setup_logging, load_modeling_data
from typing import Dict, Any
import logging
import numpy as np
from functools import reduce

def run_lss_for_subject(subject_data: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Runs the LSS modeling to estimate single-trial beta maps from one or more runs.
    """
    subject_id = subject_data['subject_id']
    bold_imgs = subject_data['bold_imgs']
    events_df = subject_data['events_df']
    confounds_dfs = subject_data['confounds_dfs']
    derivatives_dir = subject_data['derivatives_dir']
    
    logging.info(f"--- Running LSS Model for {subject_id} on {len(bold_imgs)} run(s) ---")
    
    # --- Onset Normalization ---
    corrected_events_list = []
    for run_number in sorted(events_df['run'].unique()):
        run_events_df = events_df[events_df['run'] == run_number].copy()
        if not run_events_df.empty:
            first_onset_in_run = run_events_df['onset'].min()
            run_events_df['onset'] -= first_onset_in_run
            corrected_events_list.append(run_events_df)
    
    events_df = pd.concat(corrected_events_list, ignore_index=True)

    # --- GLM Specification ---
    analysis_params = params['analysis_params']
    glm = FirstLevelModel(
        t_r=analysis_params['t_r'],
        slice_time_ref=analysis_params['slice_time_ref'],
        hrf_model=analysis_params['glm']['hrf_model'],
        drift_model='cosine',
        mask_img=subject_data['mask_file'],
        signal_scaling=False,
        n_jobs=-1
    )

    # --- Prepare a dictionary of per-run event dataframes (for nuisance regressors) ---
    events_per_run = {
        run_number: events_df[events_df['run'] == run_number].copy()
        for run_number in sorted(events_df['run'].unique())
    }
    for df in events_per_run.values():
        df['trial_type'] = 'nuisance'
        # CRITICAL FIX: Keep only essential columns for the nuisance model
        df = df[['onset', 'duration', 'trial_type']]

    # Convert to a list of dataframes in the correct order for nilearn
    initial_events_list = [events_per_run[run] for run in sorted(events_per_run.keys())]
    
    # --- Main LSS Loop ---
    all_beta_maps = []
    for trial_idx, trial in events_df.iterrows():
        logging.info(f"  - Running LSS for trial {trial_idx + 1}/{len(events_df)}")

        # Create a deep copy of the per-run events to modify for this trial
        lss_events_list = [df.copy() for df in initial_events_list]
        
        # Find which run this trial belongs to
        trial_run = trial['run']
        
        # Find the index of this run in our sorted list of runs
        run_idx = sorted(events_per_run.keys()).index(trial_run)
        
        # In the specific run's event dataframe, find the trial and mark it.
        run_specific_df = lss_events_list[run_idx]
        
        # We use np.isclose for robust floating-point comparison
        onsets_match = np.isclose(run_specific_df['onset'], trial['onset'])
        durations_match = np.isclose(run_specific_df['duration'], trial['duration'])
        
        original_trial_loc = run_specific_df[onsets_match & durations_match].index
        
        if not original_trial_loc.empty:
            # Mark the trial of interest with its correct trial_type
            # Use .loc to ensure we're modifying the actual DataFrame
            run_specific_df.loc[original_trial_loc, 'trial_type'] = trial['trial_type']
        else:
            logging.warning(f"Could not find trial {trial_idx} in its run dataframe using float comparison. Skipping.")
            continue

        # --- Fit the GLM for this single trial ---
        glm.fit(bold_imgs, events=lss_events_list, confounds=confounds_dfs)

        # --- Extract and Store the Beta Map ---
        # The contrast is simply the name of the trial_type for our trial of interest
        beta_map = glm.compute_contrast(trial['trial_type'], output_type='effect_size')
        all_beta_maps.append(beta_map)

    # Concatenate all beta maps into a single 4D NIfTI image
    beta_maps_img = image.concat_imgs(all_beta_maps)

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

    # Load the full config
    config = load_config(args.config)
    
    # Load all data for the subject using the new, definitive utility function
    subject_data = load_modeling_data(args.config, args.env, args.subject)
    
    # Pass the full config to the analysis function
    run_lss_for_subject(subject_data, config)


if __name__ == "__main__":
    main()
