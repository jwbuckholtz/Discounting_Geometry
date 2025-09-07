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

def convolve_with_hrf(events_df: pd.DataFrame, modulator: str, n_scans: int, t_r: float) -> np.ndarray:
    """Manually convolves a modulator with a canonical HRF."""
    frame_times = np.arange(n_scans) * t_r
    
    # Create the regressor time series
    regressor = np.zeros(n_scans)
    for _, trial in events_df.iterrows():
        onset_scan = int(trial['onset'] / t_r)
        # Safety check to prevent IndexError for events outside the scan time
        if onset_scan < n_scans:
            regressor[onset_scan] = trial[modulator]
        else:
            logging.warning(
                f"Trial with onset {trial['onset']} is outside the scan time "
                f"({n_scans * t_r}s). Skipping event for modulator '{modulator}'."
            )
        
    # Canonical HRF
    hrf = gamma.pdf(np.arange(0, 32, t_r), a=6, scale=1, loc=0)
    
    # Convolve and return
    return np.convolve(regressor, hrf)[:n_scans]

def run_standard_glm_for_subject(subject_data: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Runs a standard GLM with parametric modulators across one or more runs.
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
    # Fill any NaNs in the behavioral modulator columns with 0.
    # This handles cases where behavioral model fitting may have failed.
    modulator_cols = [m for m in params['glm']['contrasts'] if m in events_df.columns]
    if events_df[modulator_cols].isnull().values.any():
        logging.warning(f"NaNs found in modulator columns for {subject_id}. Filling with 0.")
        events_df[modulator_cols] = events_df[modulator_cols].fillna(0)

    # Convert categorical 'choice' column to numeric (0 or 1) for modulation
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
        smoothing_fwhm=params['smoothing_fwhm']
    )

    # --- Process Each Run Independently ---
    design_matrices = []
    for i, (bold_img, confounds_df) in enumerate(zip(bold_imgs, confounds_dfs)):
        run_number = i + 1
        logging.info(f"  - Processing run {run_number}/{len(bold_imgs)}")

        # Determine the number of scans for this specific run
        n_scans = image.load_img(bold_img).shape[3]
        
        # Isolate events for the current run
        run_events_df = events_df[events_df['run'] == run_number]

        # Use the modulators from the config file
        modulators = params['glm']['contrasts']
        
        # Create the design matrix for this run
        design_matrix = confounds_df.copy()
        if design_matrix.isnull().values.any():
            logging.warning(f"NaNs found in confounds for run {run_number}. Filling with 0.")
            design_matrix = design_matrix.fillna(0)
        
        for mod in modulators:
            if mod == 'decision':
                # Create a temporary 'decision' column for convolution for this run's events
                run_events_df_copy = run_events_df.copy()
                run_events_df_copy['decision'] = 1
                convolved_reg = convolve_with_hrf(run_events_df_copy, 'decision', n_scans, params['t_r'])
            elif mod in run_events_df.columns:
                convolved_reg = convolve_with_hrf(run_events_df, mod, n_scans, params['t_r'])
            else:
                logging.warning(f"Modulator column '{mod}' not found in events_df for run {run_number}. Skipping.")
                continue
            
            design_matrix[mod] = convolved_reg

        design_matrices.append(design_matrix)
            
    # --- Fit the GLM with all runs ---
    glm.fit(bold_imgs, design_matrices=design_matrices)

    # --- Define and Compute Contrasts ---
    # The design matrix from the last run is used to check for columns, as they are consistent.
    final_design_matrix = design_matrices[-1]
    contrasts = {c: c for c in params['glm']['contrasts'] if c in final_design_matrix.columns}

    for contrast_id, contrast_formula in contrasts.items():
        try:
            # Nilearn automatically computes contrasts across all runs
            z_map = glm.compute_contrast(contrast_formula, output_type='z_score')
            output_filename = output_dir / 'z_maps' / f"{contrast_id}_zmap.nii.gz"
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            z_map.to_filename(output_filename)
            logging.info(f"Saved z-map to {output_filename}")

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
