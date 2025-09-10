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
    Runs a standard GLM with parametric modulators across one or more runs.
    """
    subject_id = subject_data['subject_id']
    bold_imgs = subject_data['bold_imgs']
    events_df = subject_data['events_df']
    confounds_dfs = subject_data['confounds_dfs']
    derivatives_dir = subject_data['derivatives_dir']

    logging.info(f"--- Running Standard GLM for {subject_id} on {len(bold_imgs)} run(s) ---")
    output_dir = derivatives_dir / 'standard_glm' / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # CRITICAL FIX: Standardize confounds using the UNION of columns
    if confounds_dfs:
        all_confound_columns = list(reduce(set.union, [set(df.columns) for df in confounds_dfs]))
        confounds_dfs = [df.reindex(columns=all_confound_columns, fill_value=0) for df in confounds_dfs]
        logging.info(f"Standardized confounds to {len(all_confound_columns)} common columns across {len(confounds_dfs)} runs.")
    
    # --- Model Specification ---
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
    
    # --- Prepare Events for Each Run ---
    events_per_run = []
    all_modulator_cols = analysis_params['glm']['parametric_modulators']

    for i, bold_img in enumerate(bold_imgs):
        run_number = i + 1
        run_events_df = events_df[events_df['run'] == run_number].copy()

        if run_events_df.empty:
            logging.warning(f"No events found for run {run_number}. Skipping event preparation.")
            events_per_run.append(None)
            continue
        
        first_onset_in_run = run_events_df['onset'].min()
        run_events_df['onset'] -= first_onset_in_run
        run_events_df['trial_type'] = 'mean'
        
        # Ensure all potential modulator columns exist and mean-center them
        for col in all_modulator_cols:
            if col not in run_events_df.columns:
                run_events_df[col] = 0
            # Mean-center the modulator
            run_events_df[col] -= run_events_df[col].mean()
        
        nilearn_events_cols = ['onset', 'duration', 'trial_type'] + all_modulator_cols
        events_per_run.append(run_events_df[nilearn_events_cols])
            
    # --- Fit the GLM ---
    glm.fit(bold_imgs, events=events_per_run, confounds=confounds_dfs)

    # --- Define and Compute Contrasts ---
    full_design_matrix = pd.concat(glm.design_matrices_, ignore_index=True)
    contrasts_to_compute = ['mean'] + analysis_params['glm']['parametric_modulators']
    
    for contrast_name in contrasts_to_compute:
        if contrast_name == 'mean':
            contrast_cols = [col for col in full_design_matrix.columns if col.startswith('mean') and 'x' not in col]
        else:
            contrast_cols = [col for col in full_design_matrix.columns if col.startswith(f'meanx{contrast_name}')]

        if not contrast_cols:
            logging.warning(f"Contrast '{contrast_name}' not found in design matrix. Skipping.")
            continue

        contrast_vector = np.zeros(full_design_matrix.shape[1])
        weight = 1.0 / len(contrast_cols)
        for col in contrast_cols:
            contrast_vector[full_design_matrix.columns.get_loc(col)] = weight
        
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

    # Load the full config
    config = load_config(args.config)
    
    # Load all data for the subject using the utility function
    subject_data = load_concatenated_subject_data(args.config, args.env, args.subject)
    
    # Pass the full config to the analysis function
    run_standard_glm_for_subject(subject_data, config)


if __name__ == "__main__":
    main()
