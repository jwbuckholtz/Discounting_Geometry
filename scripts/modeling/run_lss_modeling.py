import os
import argparse
import yaml
import pandas as pd
import numpy as np
from glob import glob
from nilearn import image, masking
from nilearn.glm.first_level import FirstLevelModel
from pathlib import Path

from scripts.utils import load_config, find_fmriprep_files, find_behavioral_sv_file

def run_lss_modeling(subject_id, bold_file, mask_file, confounds_file, events_df):
    """
    Runs the LSS modeling to estimate single-trial beta maps.

    Args:
        subject_id (str): The subject ID.
        bold_file (str): Path to the preprocessed BOLD file.
        mask_file (str): Path to the brain mask file.
        confounds_file (str): Path to the confounds file.
        events_df (pd.DataFrame): DataFrame with event timings.

    Returns:
        Nifti1Image: A 4D Nifti image containing the beta map for each trial.
    """
    # --- 1. Load Neuroimaging Data ---
    # Get the TR from the BOLD image header and convert it to a standard Python float
    tr = image.load_img(bold_file).header.get_zooms()[-1].item()

    # Load confounds and select a subset for the model
    confounds = pd.read_csv(confounds_file, sep='\t')
    # Using a common set of confound regressors
    confound_vars = [
        'trans_x', 'trans_y', 'trans_z',
        'rot_x', 'rot_y', 'rot_z',
        'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04'
    ]
    # Fill any NaNs with the mean of the column
    confounds_selected = confounds[confound_vars].fillna(confounds[confound_vars].mean())

    # --- 2. Prepare for LSS Iteration ---
    # Add a trial_type column for nilearn, all trials are 'stim' for now
    events_df['trial_type'] = 'stim'
    beta_maps = []

    # --- 3. Run LSS Loop ---
    for i in range(len(events_df)):
        # Create the LSS events DataFrame for this trial
        lss_events = events_df.copy()
        
        # Isolate the trial of interest
        trial_of_interest = lss_events.loc[i, 'trial_type']
        lss_events.loc[i, 'trial_type'] = f'{trial_of_interest}_{i}'
        
        # All other trials are lumped into one regressor
        other_trials_mask = ~lss_events.index.isin([i])
        lss_events.loc[other_trials_mask, 'trial_type'] = 'other_trials'

        # Instantiate and fit the GLM
        glm = FirstLevelModel(t_r=tr, mask_img=mask_file,
                              standardize=True, noise_model='ar1',
                              smoothing_fwhm=5.0, high_pass=1./128,
                              n_jobs=-1) # Use all available CPUs
        
        glm.fit(bold_file, events=lss_events, confounds=confounds_selected)

        # Compute the contrast for the trial of interest
        contrast_id = f'{trial_of_interest}_{i}'
        z_map = glm.compute_contrast(contrast_id, output_type='z_score')
        beta_maps.append(z_map)
        
        print(f"  - Finished GLM for trial {i+1}/{len(events_df)}")

    # Concatenate all the single-trial beta maps into a single 4D image
    return image.concat_imgs(beta_maps)


def save_beta_maps(subject_id, beta_maps, output_dir):
    """Saves the single-trial beta maps to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{subject_id}_lss_beta_maps.nii.gz'
    beta_maps.to_filename(output_path)
    print(f"Saved beta maps for {subject_id} to {output_path}")

def main():
    """Main function to run the LSS modeling."""
    parser = argparse.ArgumentParser(description="Run LSS modeling to generate single-trial beta maps.")
    parser.add_argument('--config', default='config/project_config.yaml', help='Path to the project configuration file.')
    parser.add_argument('--env', default='local', choices=['local', 'hpc'], help="Environment from the config file.")
    parser.add_argument('--subject', required=True, help='Subject ID (e.g., sub-s061).')
    args = parser.parse_args()

    # 1. Load configuration and get paths
    config = load_config(args.config)
    env_config = config[args.env]
    derivatives_dir = Path(env_config['derivatives_dir'])
    fmriprep_dir = Path(env_config['fmriprep_dir'])

    # 2. Find fMRIPrep files
    print("Finding fMRIPrep files...")
    bold_file, mask_file, confounds_file = find_fmriprep_files(fmriprep_dir, args.subject)
    
    # 3. Load behavioral data
    print("Loading behavioral data...")
    events_file = find_behavioral_sv_file(derivatives_dir, args.subject)
    events_df = pd.read_csv(events_file, sep='\t')

    # 4. Run the LSS model
    print(f"Starting LSS modeling for {args.subject}...")
    beta_maps = run_lss_modeling(args.subject, bold_file, mask_file, confounds_file, events_df)

    # 5. Save the resulting beta maps
    output_dir = derivatives_dir / 'lss_betas' / args.subject
    save_beta_maps(args.subject, beta_maps, output_dir)
    
    print(f"Successfully finished LSS modeling for {args.subject}")


if __name__ == '__main__':
    main()
