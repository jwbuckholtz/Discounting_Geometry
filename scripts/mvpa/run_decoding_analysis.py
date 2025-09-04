import os
import argparse
import yaml
import pandas as pd
import numpy as np
from glob import glob
from nilearn import image
from nilearn.decoding import Decoder
from sklearn.model_selection import StratifiedKFold, KFold
from pathlib import Path
import warnings
from nilearn.maskers import NiftiMasker

from scripts.utils import load_config, find_fmriprep_files, find_lss_beta_maps, find_behavioral_sv_file

def resample_roi_to_betas(roi_path, beta_maps_path):
    """Resamples an ROI mask to the space of the beta maps."""
    print(f"Resampling ROI mask to match beta maps space...")
    # Use the first volume of the beta maps as the reference for resampling
    target_affine = image.index_img(beta_maps_path, 0).affine
    target_shape = image.index_img(beta_maps_path, 0).shape

    resampled_roi = image.resample_to_img(
        source_img=image.load_img(str(roi_path)),
        target_img=image.index_img(beta_maps_path, 0),
        interpolation='nearest'
    )
    return resampled_roi

def load_data(subject_id, derivatives_dir):
    """
    Loads the necessary data for a single subject's decoding analysis.
    
    This function will load:
    1. The single-trial beta maps.
    2. The behavioral data with subjective values.

    Args:
        subject_id (str): The ID of the subject.
        derivatives_dir (str): Path to the project's derivatives directory.

    Returns:
        tuple: A tuple containing the beta maps Nifti image and the events DataFrame.
    """
    # 1. Find and load the single-trial beta maps
    betas_path = find_lss_beta_maps(derivatives_dir, subject_id)
    beta_maps_img = image.load_img(str(betas_path))

    # 2. Find and load the processed behavioral data
    events_path = find_behavioral_sv_file(derivatives_dir, subject_id)
    events_df = pd.read_csv(events_path, sep='\t')

    return beta_maps_img, events_df

def prepare_decoding_data(events_df, target_variable, n_betas):
    """
    Prepares the data for the decoding analysis by creating the labels (y) and
    identifying the relevant trials. This function is robust to NaNs.

    Args:
        events_df (pd.DataFrame): The DataFrame of events.
        target_variable (str): The name of the column to be used as the decoding target.
        n_betas (int): The number of beta maps, for validation.

    Returns:
        tuple: A tuple containing the labels (y) as a NumPy array and a boolean
               mask for the trials to include.
    """
    # --- 1. Validation ---
    if target_variable not in events_df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in events DataFrame.")
    
    if len(events_df) != n_betas:
        raise ValueError(f"Mismatch between number of events ({len(events_df)}) and beta maps ({n_betas}).")

    # --- 2. Create Labels and Trial Mask based on NaNs ---
    labels_series = events_df[target_variable].copy()
    valid_trials_mask = labels_series.notna().values
    
    # --- 3. Handle Categorical vs. Continuous Targets ---
    if labels_series.dtype == 'object':
        # For categorical data, factorize only the non-NaN values
        valid_labels = labels_series[valid_trials_mask]
        codes, uniques = pd.factorize(valid_labels)
        
        print(f"Found {len(uniques)} unique labels for categorical target '{target_variable}': {uniques.tolist()}")
        
        # Create a new float array to store codes, preserving NaNs elsewhere
        final_labels = np.full(labels_series.shape, np.nan)
        final_labels[valid_trials_mask] = codes
    else:
        # For numeric data, just use the values
        final_labels = labels_series.values

    return final_labels, valid_trials_mask


def run_decoding(beta_maps_img, mask_img, labels, valid_trials_mask, is_categorical):
    """
    Runs the MVPA/decoding analysis using scikit-learn directly for robustness.

    Args:
        beta_maps_img: The 4D Nifti image of single-trial beta maps.
        mask_img: The brain mask Nifti image.
        labels (np.array): The labels for each trial.
        valid_trials_mask (np.array): A boolean mask of valid trials.
        is_categorical (bool): True if the target is categorical (classification).

    Returns:
        np.array: An array of cross-validation scores.
    """
    # --- 1. Filter Data and Apply Mask ---
    fmri_data_valid = image.index_img(beta_maps_img, valid_trials_mask)
    labels_valid = labels[valid_trials_mask]

    # Use NiftiMasker to extract the time series from the ROIs
    masker = NiftiMasker(mask_img=mask_img, standardize=True)
    X = masker.fit_transform(fmri_data_valid)
    y = labels_valid

    # --- 2. Set up the Model and Cross-Validation ---
    if is_categorical:
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        print("Running classification analysis...")
        model = make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced'))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'accuracy'
    else:
        from sklearn.svm import SVR
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        print("Running regression analysis...")
        model = make_pipeline(StandardScaler(), SVR(kernel='linear'))
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'r2'
        
    # --- 3. Run Cross-Validation ---
    from sklearn.model_selection import cross_val_score
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    return scores


def save_results(subject_id, target_variable, scores, output_dir, roi_name):
    """
    Saves the decoding results to a file.

    Args:
        subject_id (str): The subject ID.
        target_variable (str): The decoded target variable.
        scores (np.array): The cross-validation scores.
        output_dir (str): The directory to save the results in.
        roi_name (str): The name of the ROI used.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({'scores': scores})
    results_df['fold'] = range(1, len(scores) + 1)
    
    output_filename = f'{subject_id}_target-{target_variable}_roi-{roi_name}_decoding-scores.tsv'
    output_path = output_dir / output_filename
    results_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Saved decoding scores to {output_path}")
    print(f"Mean score for ROI '{roi_name}': {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")


def main():
    """Main function to run the decoding analysis."""
    parser = argparse.ArgumentParser(description="Run MVPA/decoding analysis.")
    parser.add_argument('--config', default='config/project_config.yaml', help='Path to the project configuration file.')
    parser.add_argument('--env', default='local', choices=['local', 'hpc'], help="Environment from the config file.")
    parser.add_argument('--subject', required=True, help='Subject ID (e.g., sub-s061).')
    parser.add_argument('--target', required=True, help='The variable to decode (e.g., choice, SVchosen).')
    parser.add_argument('--roi-path', help='Path to a specific ROI mask file or a directory of ROI masks.')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    env_config = config[args.env]
    derivatives_dir = Path(env_config['derivatives_dir'])
    fmriprep_dir = Path(env_config['fmriprep_dir'])

    # 1. Load data (betas and events)
    print("Loading beta maps and behavioral data...")
    beta_maps_img, events_df = load_data(args.subject, derivatives_dir)

    # 2. Prepare data for decoding (this is independent of the mask)
    print(f"Preparing data for decoding target: {args.target}")
    labels, valid_trials_mask = prepare_decoding_data(events_df, args.target, beta_maps_img.shape[-1])
    is_categorical = (events_df[args.target].dtype == 'object')

    # 3. Determine which mask(s) to use
    if args.roi_path:
        roi_path = Path(args.roi_path)
        if roi_path.is_dir():
            roi_files = sorted(list(roi_path.glob('*.nii.gz')) + list(roi_path.glob('*.nii')))
            print(f"Found {len(roi_files)} ROI files in directory: {roi_path}")
        elif roi_path.is_file():
            roi_files = [roi_path]
        else:
            raise FileNotFoundError(f"ROI path not found: {args.roi_path}")
    else:
        # If no ROI path is provided, use the whole-brain mask as a single "ROI"
        print("No ROI path provided. Using whole-brain mask.")
        # Find the whole-brain mask which we'll use as a default
        # The find_fmriprep_files function is robust to session variability.
        _, brain_mask_path, _ = find_fmriprep_files(fmriprep_dir, args.subject)
        roi_files = [Path(brain_mask_path)]

    # 4. Loop through each mask, run decoding, and save results
    for roi_file in roi_files:
        roi_name = roi_file.name.split('.')[0] # Get a clean name, e.g., 'vmPFC' from 'vmPFC.nii.gz'
        print(f"\n--- Running analysis for ROI: {roi_name} ---")
        
        mask_img_orig = image.load_img(str(roi_file))
        
        # Resample the mask to match the beta maps' space
        mask_img = resample_roi_to_betas(roi_file, beta_maps_img)

        # Run the decoding with the current mask
        scores = run_decoding(beta_maps_img, mask_img, labels, valid_trials_mask, is_categorical)

        # Save the results for the current mask
        output_dir = derivatives_dir / 'mvpa' / args.subject
        save_results(args.subject, args.target, scores, output_dir, roi_name)

    print(f"\nSuccessfully finished all analyses for {args.subject}, target: {args.target}")

if __name__ == '__main__':
    main()
