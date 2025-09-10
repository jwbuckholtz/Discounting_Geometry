import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from scripts.utils import load_modeling_data, load_config, setup_logging
from typing import Dict, Any
import logging
from nilearn import image
from functools import reduce

def _validate_modulators_across_runs(events_df: pd.DataFrame, all_modulator_cols: list, run_numbers: list) -> list:
    """
    Validates modulators across all runs and returns only those that are valid everywhere.
    This ensures consistent design matrices across runs.
    
    Args:
        events_df: Complete events DataFrame with all runs
        all_modulator_cols: List of potential modulator columns
        run_numbers: List of run numbers to check
        
    Returns:
        List of modulator columns that are valid across all runs
    """
    valid_modulators = []
    
    for modulator in all_modulator_cols:
        is_valid_across_runs = True
        
        for run_number in run_numbers:
            run_events = events_df[events_df['run'] == run_number]
            
            # Check if column exists and handle NaN values
            if modulator not in run_events.columns:
                run_events_copy = run_events.copy()
                run_events_copy[modulator] = 0
            else:
                run_events_copy = run_events.copy()
                
            # Handle all-NaN columns safely
            if run_events_copy[modulator].isna().all():
                logging.warning(f"Modulator '{modulator}' is all NaN in run {run_number}. Will be excluded.")
                is_valid_across_runs = False
                break
                
            # Fill remaining NaNs with column mean and center
            col_mean = run_events_copy[modulator].mean()
            if pd.isna(col_mean):
                logging.warning(f"Modulator '{modulator}' has no valid values in run {run_number}. Will be excluded.")
                is_valid_across_runs = False
                break
                
            run_events_copy[modulator] = run_events_copy[modulator].fillna(col_mean)
            run_events_copy[modulator] -= col_mean
            
            # Check for zero variance after centering
            if np.isclose(run_events_copy[modulator].var(), 0):
                logging.warning(f"Modulator '{modulator}' has zero variance in run {run_number}. Will be excluded.")
                is_valid_across_runs = False
                break
                
        if is_valid_across_runs:
            valid_modulators.append(modulator)
            
    return valid_modulators

def prepare_run_events(run_events_df: pd.DataFrame, valid_modulator_cols: list) -> pd.DataFrame:
    """
    Prepares a single run's event DataFrame for the GLM in the format Nilearn expects
    for parametric modulation. Only processes modulators that have been validated across all runs.
    """
    # Clean and mean-center the validated modulator columns
    for col in valid_modulator_cols:
        if col not in run_events_df.columns:
            run_events_df[col] = 0
        else:
            # Handle NaN values safely (should be rare after validation)
            col_mean = run_events_df[col].mean()
            if not pd.isna(col_mean):
                run_events_df[col] = run_events_df[col].fillna(col_mean)
                run_events_df[col] -= col_mean
            else:
                # Fallback: set to zero if still all NaN
                run_events_df[col] = 0

    # Base events for the main effect of each trial
    events_base = run_events_df[['onset', 'duration']].copy()
    events_base['trial_type'] = 'mean'
    events_base['modulation'] = 1
    
    # Create new event rows for each validated parametric modulator
    events_modulated_list = [events_base]
    for modulator in valid_modulator_cols:
        modulator_events = run_events_df[['onset', 'duration']].copy()
        modulator_events['trial_type'] = modulator
        modulator_events['modulation'] = run_events_df[modulator]
        events_modulated_list.append(modulator_events)

    # Combine all event types into a single long-format DataFrame
    final_events_df = pd.concat(events_modulated_list, ignore_index=True)
    
    return final_events_df

def run_standard_glm_for_subject(subject_data: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Runs a standard GLM with parametric modulators across one or more runs.
    """
    subject_id = subject_data['subject_id']
    bold_imgs = subject_data['bold_imgs']
    run_numbers = subject_data['run_numbers']
    events_df = subject_data['events_df']
    confounds_dfs = subject_data['confounds_dfs']
    derivatives_dir = subject_data['derivatives_dir']

    logging.info(f"--- Running Standard GLM for {subject_id} on {len(bold_imgs)} run(s) ---")
    output_dir = derivatives_dir / 'standard_glm' / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_params = params['analysis_params']
    glm = FirstLevelModel(
        t_r=analysis_params['t_r'],
        slice_time_ref=analysis_params['slice_time_ref'],
        hrf_model=analysis_params['glm']['hrf_model'],
        drift_model=analysis_params['glm']['drift_model'],
        smoothing_fwhm=analysis_params['smoothing_fwhm'],
        mask_img=subject_data['mask_file'],
        n_jobs=-1
    )
    
    # --- Validate Modulators Across All Runs ---
    all_modulator_cols = analysis_params['glm']['parametric_modulators']
    valid_modulator_cols = _validate_modulators_across_runs(events_df, all_modulator_cols, run_numbers)
    
    if len(valid_modulator_cols) < len(all_modulator_cols):
        dropped = set(all_modulator_cols) - set(valid_modulator_cols)
        logging.info(f"Dropped modulators that were invalid across runs: {dropped}")
    
    # --- Prepare Events for Each Run ---
    events_per_run = []
    valid_bold_imgs = []
    valid_confounds_dfs = []
    valid_run_numbers = []
    
    for i, run_number in enumerate(run_numbers):
        run_events_df = events_df[events_df['run'] == run_number].copy()
        
        # Skip runs with no events
        if run_events_df.empty:
            logging.warning(f"Skipping run {run_number} - no events found")
            continue
            
        # Check if onsets need normalization (only if not already run-relative)
        first_onset = run_events_df['onset'].min()
        if first_onset > 10:  # Heuristic: if first onset > 10s, likely session-relative
            logging.info(f"Normalizing onsets for run {run_number} (first onset: {first_onset:.2f}s)")
            run_events_df['onset'] -= first_onset
        else:
            logging.info(f"Onsets appear already run-relative for run {run_number} (first onset: {first_onset:.2f}s)")
        
        prepared_events = prepare_run_events(run_events_df, valid_modulator_cols)
        events_per_run.append(prepared_events)
        valid_bold_imgs.append(bold_imgs[i])
        valid_confounds_dfs.append(confounds_dfs[i])
        valid_run_numbers.append(run_number)
    
    # Check if we have any valid runs left
    if not events_per_run:
        raise ValueError(f"No valid runs with events found for subject {subject_id}")
        
    logging.info(f"Processing {len(events_per_run)} valid runs (skipped {len(run_numbers) - len(events_per_run)} empty runs)")
            
    # --- Fit the GLM ---
    glm.fit(valid_bold_imgs, events=events_per_run, confounds=valid_confounds_dfs)
    
    # --- Verify Design Matrix Consistency ---
    if len(glm.design_matrices_) > 1:
        # Check that all design matrices have the same columns
        first_columns = set(glm.design_matrices_[0].columns)
        for i, design_matrix in enumerate(glm.design_matrices_[1:], 1):
            current_columns = set(design_matrix.columns)
            if current_columns != first_columns:
                missing = first_columns - current_columns
                extra = current_columns - first_columns
                logging.warning(f"Design matrix inconsistency in run {valid_run_numbers[i]}:")
                if missing:
                    logging.warning(f"  Missing columns: {missing}")
                if extra:
                    logging.warning(f"  Extra columns: {extra}")
                    
        # Log the final design matrix structure
        logging.info(f"Design matrices have {len(first_columns)} columns: {sorted(first_columns)}")

    # --- Define and Compute Contrasts ---
    # Check for contrasts across ALL design matrices, not just the first one
    all_design_columns = set()
    for design_matrix in glm.design_matrices_:
        all_design_columns.update(design_matrix.columns)
    
    contrasts_to_compute = ['mean'] + valid_modulator_cols
    
    for contrast_name in contrasts_to_compute:
        if contrast_name not in all_design_columns:
            logging.warning(
                f"Contrast '{contrast_name}' not found in any design matrix. Skipping."
            )
            continue
            
        # Check if the contrast exists in ALL runs (required for valid contrast)
        missing_in_runs = []
        for i, design_matrix in enumerate(glm.design_matrices_):
            if contrast_name not in design_matrix.columns:
                missing_in_runs.append(i + 1)
                
        if missing_in_runs:
            logging.warning(
                f"Contrast '{contrast_name}' missing in run(s) {missing_in_runs}. Skipping."
            )
            continue
            
        # The regressor names are now directly the trial_type names
        z_map = glm.compute_contrast(contrast_name, output_type='z_score')
        
        contrast_filename = output_dir / f"contrast-{contrast_name}_zmap.nii.gz"
        z_map.to_filename(contrast_filename)
        logging.info(f"Saved z-map to: {contrast_filename.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standard GLM for a single subject.")
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
    run_standard_glm_for_subject(subject_data, config)


if __name__ == "__main__":
    main()
