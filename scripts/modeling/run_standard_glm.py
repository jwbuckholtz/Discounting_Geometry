import argparse
from pathlib import Path
from nilearn.glm.first_level import FirstLevelModel
from scripts.utils import load_concatenated_subject_data

def run_standard_glm_for_subject(subject_data):
    """
    Runs a standard GLM with parametric modulators across one or more runs.
    """
    subject_id = subject_data['subject_id']
    bold_imgs = subject_data['bold_imgs']
    mask_file = subject_data['mask_file']
    events_df = subject_data['events_df']
    confounds_dfs = subject_data['confounds_dfs']
    derivatives_dir = subject_data['derivatives_dir']

    print(f"--- Running Standard GLM for {subject_id} on {len(bold_imgs)} run(s) ---")
    output_dir = derivatives_dir / 'first_level_glms' / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use all trials in the events dataframe, which now includes a 'run' column
    events_df['trial_type'] = 'decision'
    
    # Define the GLM
    glm = FirstLevelModel(
        t_r=2.0,
        slice_time_ref=0.5,
        hrf_model='glover',
        drift_model='cosine',
        mask_img=mask_file,
        signal_scaling=False,
        smoothing_fwhm=5.0
    )

    # Fit the GLM across all runs
    glm.fit(bold_imgs, events=events_df, confounds=confounds_dfs)

    # --- Define and Compute Contrasts ---
    contrasts = {
        'choice': 'decision*choice',
        'SVchosen': 'decision*SVchosen',
        'SVunchosen': 'decision*SVunchosen',
        'SVsum': 'decision*SVsum',
        'SVdiff': 'decision*SVdiff'
    }

    for contrast_id, contrast_formula in contrasts.items():
        print(f"Computing contrast: {contrast_id}")
        contrast_map = glm.compute_contrast(contrast_formula, output_type='effect_size')
        
        contrast_filename = output_dir / f"{subject_id}_contrast-{contrast_id}_map.nii.gz"
        contrast_map.to_filename(contrast_filename)
        print(f"Saved contrast map to {contrast_filename}")


def main():
    parser = argparse.ArgumentParser(description="Run standard GLM for a single subject.")
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to the project config file')
    parser.add_argument('--env', type=str, required=True, choices=['local', 'hpc'], help='Environment to run on')
    parser.add_argument('--subject', type=str, required=True, help='The subject ID to process')
    args = parser.parse_args()

    # --- Load Data ---
    subject_data = load_concatenated_subject_data(args.config, args.env, args.subject)
    
    # --- Run Analysis ---
    run_standard_glm_for_subject(subject_data)


if __name__ == "__main__":
    main()
