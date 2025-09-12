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
    # CRITICAL FIX: Safe access to analysis_params with informative error
    analysis_params = params.get('analysis_params')
    if analysis_params is None:
        raise ValueError(f"Missing 'analysis_params' in params for LSS. "
                        f"Available keys: {list(params.keys())}")
    
    subject_id = subject_data['subject_id']
    bold_imgs = subject_data['bold_imgs']
    events_df = subject_data['events_df']
    confounds_dfs = subject_data['confounds_dfs']
    derivatives_dir = subject_data['derivatives_dir']
    
    logging.info(f"--- Running LSS Model for {subject_id} on {len(bold_imgs)} run(s) ---")
    
    # CRITICAL FIX: Validate required columns before processing
    required_event_columns = ['onset', 'duration', 'run', 'trial_type']
    missing_columns = [col for col in required_event_columns if col not in events_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required event columns for LSS: {missing_columns}. "
                        f"Available columns: {list(events_df.columns)}")
    
    logging.info(f"LSS event validation passed - required columns present: {required_event_columns}")
    
    # CRITICAL FIX: Ensure consistent data types for run comparisons
    # Convert run column to integer for consistent type matching
    try:
        events_df['run'] = events_df['run'].astype(int)
        logging.info("Converted LSS events run column to integer for consistent type matching")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot convert LSS events run column to integer: {e}. "
                        f"Run values: {events_df['run'].unique()}")
    
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
    
    # CRITICAL FIX: Intelligent CPU allocation for HPC environments
    # Avoid oversubscribing CPUs beyond SLURM allocation
    import os
    n_jobs = -1  # Default: use all cores
    
    # Check for SLURM environment and respect CPU allocation
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        try:
            n_jobs = int(slurm_cpus)
            # CRITICAL FIX: Validate n_jobs is positive to prevent FirstLevelModel crashes
            if n_jobs <= 0:
                logging.warning(f"Invalid SLURM_CPUS_PER_TASK={n_jobs} (must be >= 1), using default n_jobs=-1")
                n_jobs = -1
            else:
                logging.info(f"Using SLURM_CPUS_PER_TASK={n_jobs} for LSS parallel processing")
        except (ValueError, TypeError):
            logging.warning(f"Invalid SLURM_CPUS_PER_TASK value: {slurm_cpus}, using default n_jobs=-1")
    
    # Allow configuration override with type checking
    config_n_jobs = analysis_params.get('n_jobs')
    if config_n_jobs is not None:
        try:
            n_jobs = int(config_n_jobs)
            # CRITICAL FIX: Validate n_jobs is positive to prevent FirstLevelModel crashes
            if n_jobs <= 0:
                logging.warning(f"Invalid n_jobs configuration={n_jobs} (must be >= 1), resetting to n_jobs=1")
                logging.warning("n_jobs must be a positive integer value (>= 1)")
                n_jobs = 1  # Safe default for single-threaded processing
            else:
                logging.info(f"Using configured n_jobs={n_jobs} for LSS parallel processing")
        except (ValueError, TypeError):
            logging.warning(f"Invalid n_jobs configuration value: {config_n_jobs} (type: {type(config_n_jobs)}), using current n_jobs={n_jobs}")
            logging.warning("n_jobs must be an integer value")
    
    # CRITICAL FIX: Safe parameter lookups with informative error messages
    glm_params = analysis_params.get('glm', {})
    
    # CRITICAL FIX: Handle explicit null values in config by falling back to defaults
    hrf_model = glm_params.get('hrf_model', 'glover')
    if hrf_model is None:
        hrf_model = 'glover'
        logging.info("hrf_model is null in config - using default 'glover'")
    
    # Required analysis parameters 
    required_analysis_params = ['t_r', 'slice_time_ref']
    missing_params = [param for param in required_analysis_params if param not in analysis_params]
    if missing_params:
        raise ValueError(f"Missing required LSS analysis parameters: {missing_params}. "
                        f"Available parameters: {list(analysis_params.keys())}")
    
    glm = FirstLevelModel(
        t_r=analysis_params['t_r'],
        slice_time_ref=analysis_params['slice_time_ref'],
        hrf_model=hrf_model,
        drift_model='cosine',  # Fixed for LSS
        mask_img=subject_data['mask_file'],
        signal_scaling=False,
        n_jobs=n_jobs
    )
    
    logging.info(f"LSS GLM configured: HRF={hrf_model}, TR={analysis_params['t_r']}s, n_jobs={n_jobs}")

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

    # CRITICAL FIX: Ensure derivatives_dir is a Path object to prevent TypeError on path arithmetic
    derivatives_path = Path(derivatives_dir)
    
    # Save the beta maps NIfTI image
    output_dir = derivatives_path / 'lss_betas' / subject_id
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
