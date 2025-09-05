import argparse
from pathlib import Path
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image
from scripts.utils import load_concatenated_subject_data

def run_lss_for_subject(subject_data):
    """
    Runs the LSS modeling to estimate single-trial beta maps from one or more runs.
    """
    subject_id = subject_data['subject_id']
    bold_imgs = subject_data['bold_imgs']
    mask_file = subject_data['mask_file']
    events_df = subject_data['events_df']
    confounds_dfs = subject_data['confounds_dfs']
    derivatives_dir = subject_data['derivatives_dir']

    print(f"--- Running LSS Model for {subject_id} on {len(bold_imgs)} run(s) ---")

    # Define the GLM
    glm = FirstLevelModel(
        t_r=2.0,
        slice_time_ref=0.5,
        hrf_model='glover',
        drift_model='cosine',
        mask_img=mask_file,
        signal_scaling=False,
    )

    # --- LSS Modeling ---
    # The events dataframe already contains all trials from all runs
    lss_events_df = events_df[['onset', 'duration']].copy()

    beta_maps = []
    for i in range(len(lss_events_df)):
        # Create a design matrix for the current trial vs. all others
        lss_events = lss_events_df.copy()
        # Create a trial_type column for this specific iteration
        # All trials are 'other' by default
        lss_events['trial_type'] = 'other'
        # Set the trial of interest to 'target'
        lss_events.loc[i, 'trial_type'] = 'target'
        
        # Fit GLM for the current trial across all runs
        glm.fit(bold_imgs, events=lss_events, confounds=confounds_dfs)

        # Get the beta map for the target trial
        beta_map = glm.compute_contrast('target', output_type='effect_size')
        beta_maps.append(beta_map)
        if (i+1) % 10 == 0:
            print(f"  - Completed LSS for trial {i+1}/{len(lss_events_df)}")


    # Concatenate all beta maps into a single 4D NIfTI image
    beta_maps_img = image.concat_imgs(beta_maps)

    # Save the beta maps NIfTI image
    output_dir = derivatives_dir / 'lss_betas' / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{subject_id}_lss_beta_maps.nii.gz"
    beta_maps_img.to_filename(output_filename)
    print(f"LSS beta maps saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Run LSS modeling for a single subject.")
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to the project config file')
    parser.add_argument('--env', type=str, required=True, choices=['local', 'hpc'], help='Environment to run on')
    parser.add_argument('--subject', type=str, required=True, help='The subject ID to process')
    args = parser.parse_args()

    # --- Load Data ---
    # This function now loads and prepares data from all runs for the subject
    subject_data = load_concatenated_subject_data(args.config, args.env, args.subject)
    
    # --- Run Analysis ---
    run_lss_for_subject(subject_data)


if __name__ == "__main__":
    main()
