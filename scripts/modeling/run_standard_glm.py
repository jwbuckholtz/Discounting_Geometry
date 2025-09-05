import argparse
from pathlib import Path
from nilearn.glm.first_level import FirstLevelModel
from scripts.utils import load_concatenated_subject_data, load_config, setup_logging
from typing import Dict, Any
import logging

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
    output_dir = derivatives_dir / 'first_level_glms' / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add the parametric modulators directly to the events dataframe.
    # The 'trial_type' column tells the GLM which events to modulate.
    events_df['trial_type'] = 'decision'
    
    # Ensure modulator columns exist for nilearn to use them
    modulators = ['choice', 'SVchosen', 'SVunchosen', 'SVsum', 'SVdiff']
    for mod in modulators:
        if mod not in events_df.columns:
            raise ValueError(f"Modulator column '{mod}' not found in events dataframe.")
    
    # Define the GLM using parameters from the config file
    glm = FirstLevelModel(
        t_r=params['t_r'],
        slice_time_ref=params['slice_time_ref'],
        hrf_model='glover',
        drift_model='cosine',
        mask_img=mask_file,
        signal_scaling=False,
        smoothing_fwhm=params['smoothing_fwhm']
    )

    # Fit the GLM across all runs
    glm.fit(bold_imgs, events=events_df, confounds=confounds_dfs)

    # --- Define and Compute Contrasts ---
    # With parametric modulation, the contrast is simply the name of the column.
    # We also add the main 'decision' effect.
    contrasts = {
        'decision': 'decision',
        'choice': 'choice',
        'SVchosen': 'SVchosen',
        'SVunchosen': 'SVunchosen',
        'SVsum': 'SVsum',
        'SVdiff': 'SVdiff'
    }

    for contrast_id, contrast_formula in contrasts.items():
        logging.info(f"Computing contrast: {contrast_id}")
        contrast_map = glm.compute_contrast(contrast_formula, output_type='effect_size')
        
        contrast_filename = output_dir / f"{subject_id}_contrast-{contrast_id}_map.nii.gz"
        contrast_map.to_filename(contrast_filename)
        logging.info(f"Saved contrast map to {contrast_filename}")


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
