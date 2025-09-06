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
        regressor[onset_scan] = trial[modulator]
        
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

    # Convert categorical 'choice' column to numeric (0 or 1) for modulation
    # This must be done before the convolution loop.
    if 'choice' in events_df.columns:
        events_df['choice'] = (events_df['choice'] == 'larger_later').astype(int)

    # --- Manually Create Parametric Regressors ---
    # Determine the number of scans from the first run's BOLD image
    try:
        n_scans = image.load_img(bold_imgs[0]).shape[3]
    except Exception as e:
        logging.error(f"Could not determine number of scans from bold_imgs: {e}")
        raise

    # Use the modulators from the config file
    modulators = params['glm']['contrasts']
    
    # Create a copy of the nuisance regressors to build our design matrix
    design_matrix = confounds_dfs[0].copy()
    
    for mod in modulators:
        # For the main 'decision' event, we convolve a vector of ones
        if mod == 'decision':
            events_df['decision'] = 1 
        
        if mod in events_df.columns:
            convolved_reg = convolve_with_hrf(events_df, mod, n_scans, params['t_r'])
            design_matrix[mod] = convolved_reg
        else:
            logging.warning(f"Modulator column '{mod}' not found in events_df. Skipping.")
            
    glm = FirstLevelModel(
        t_r=params['t_r'],
        slice_time_ref=params['slice_time_ref'],
        hrf_model=params['glm']['hrf_model'],
        drift_model=params['glm']['drift_model'],
        mask_img=mask_file,
        signal_scaling=False,
        smoothing_fwhm=params['smoothing_fwhm']
    )

    # All regressors are now in the design_matrix, so we pass it directly.
    # nilearn ignores 'events' and 'confounds' when 'design_matrices' is provided.
    glm.fit(bold_imgs, design_matrices=[design_matrix])

    # --- Define and Compute Contrasts ---
    contrasts = {c: c for c in modulators if c in design_matrix.columns}

    for contrast_id, contrast_formula in contrasts.items():
        try:
            contrast_map = glm.compute_contrast(contrast_formula, output_type='effect_size')
            contrast_filename = output_dir / f"{subject_id}_contrast-{contrast_id}_map.nii.gz"
            contrast_map.to_filename(contrast_filename)
            logging.info(f"Saved contrast map to {contrast_filename}")
        except ValueError:
            logging.warning(f"Contrast '{contrast_id}' not found. Skipping.")


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
