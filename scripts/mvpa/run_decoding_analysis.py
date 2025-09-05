import os
import argparse
import yaml
import pandas as pd
import numpy as np
from glob import glob
from nilearn import image
from nilearn.decoding import Decoder
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold, GroupKFold
from pathlib import Path
import warnings
from nilearn.maskers import NiftiMasker
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from typing import Tuple, Dict, Any, List
import nibabel as nib
import logging

from scripts.utils import load_config, find_fmriprep_files, find_lss_beta_maps, find_behavioral_sv_file, setup_logging

def resample_roi_to_betas(roi_path: Path, beta_maps_path: Path) -> nib.Nifti1Image:
    """Resamples an ROI mask to the space of the beta maps."""
    logging.info(f"Resampling ROI mask to match beta maps space...")
    
    resampled_roi = image.resample_to_img(
        source_img=image.load_img(str(roi_path)),
        target_img=image.index_img(str(beta_maps_path), 0),
        interpolation='nearest'
    )
    return resampled_roi

def load_data(subject_id: str, derivatives_dir: Path) -> Tuple[nib.Nifti1Image, pd.DataFrame]:
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

def prepare_decoding_data(events_df: pd.DataFrame, target_variable: str, n_betas: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        
        logging.info(f"Found {len(uniques)} unique labels for categorical target '{target_variable}': {uniques.tolist()}")
        
        # Create a new float array to store codes, preserving NaNs elsewhere
        final_labels = np.full(labels_series.shape, np.nan)
        final_labels[valid_trials_mask] = codes
    else:
        # For numeric data, just use the values
        final_labels = labels_series.values

    return final_labels, valid_trials_mask, groups


def run_decoding(beta_maps_img: nib.Nifti1Image, mask_img: nib.Nifti1Image, labels: np.ndarray, valid_trials_mask: np.ndarray, groups: np.ndarray, is_categorical: bool, cv_params: Dict[str, Any]) -> np.ndarray:
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
    groups_valid = groups[valid_trials_mask]

    # Use NiftiMasker to extract the time series from the ROIs
    # standardize=False is crucial to prevent data leakage across CV folds.
    # Standardization is correctly handled within the scikit-learn pipeline.
    masker = NiftiMasker(mask_img=mask_img, standardize=False)
    X = masker.fit_transform(fmri_data_valid)
    y = labels_valid

    # --- 2. Set up the Model and Cross-Validation ---
    if is_categorical:
        logging.info("Running classification analysis with StratifiedGroupKFold...")
        model = make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced'))
        cv = StratifiedGroupKFold(n_splits=cv_params['n_splits'], shuffle=True, random_state=cv_params['random_state'])
        scoring = 'accuracy'
    else:
        logging.info("Running regression analysis with GroupKFold...")
        model = make_pipeline(StandardScaler(), SVR(kernel='linear'))
        cv = GroupKFold(n_splits=cv_params['n_splits'])
        scoring = 'r2'
        
    # --- 3. Run Cross-Validation ---
    # Pass the groups to the cv iterator
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, groups=groups_valid)
    
    return scores


def save_results(subject_id: str, target_variable: str, scores: np.ndarray, output_dir: Path, roi_name: str) -> None:
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
    
    logging.info(f"Saved decoding scores to {output_path}")
    logging.info(f"Mean score for ROI '{roi_name}': {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")


def run_subject_level_decoding(subject_id: str, derivatives_dir: Path, target: str, params: Dict[str, Any]):
    """
    Runs the decoding analysis for a single subject and a single target variable.
    """
    setup_logging()
    logging.info(f"Running decoding for sub-{subject_id}, target: {target}")

    # --- 1. Load Data ---
    lss_betas_path = find_lss_beta_maps(derivatives_dir, subject_id)
    events_path = find_behavioral_sv_file(derivatives_dir, subject_id)
    beta_maps_img = image.load_img(lss_betas_path)
    events_df = pd.read_csv(events_path, sep='\t')
    
    # --- 2. Determine Analysis Type & Get Parameters ---
    if target not in events_df.columns:
        raise ValueError(f"Target '{target}' not found in events file.")
    
    is_categorical = events_df[target].dtype == 'object' or events_df[target].nunique() < 3
    
    if is_categorical:
        analysis_params = params['mvpa']['classification']
        model = SVC(kernel='linear', class_weight='balanced')
        logging.info(f"Treating '{target}' as a CLASSIFICATION target.")
    else:
        analysis_params = params['mvpa']['regression']
        model = SVR(kernel='linear')
        logging.info(f"Treating '{target}' as a REGRESSION target.")
        
    # --- 3. Prepare Data ---
    y = events_df[target].values
    groups = events_df['run'].values if 'run' in events_df.columns else None
    valid_trials = ~pd.isna(y)
    
    X_img = image.index_img(beta_maps_img, valid_trials)
    y = y[valid_trials]
    if groups is not None:
        groups = groups[valid_trials]

    # --- 4. Run Decoding ---
    # We use the whole-brain mask for this simplified script
    _, mask_path, _ = find_fmriprep_files(derivatives_dir / 'fmriprep', subject_id)
    
    decoder = Decoder(
        estimator=model,
        mask=mask_path,
        standardize=True,
        scoring=analysis_params['scoring'],
        cv=analysis_params['cv_folds']
    )
    decoder.fit(X_img, y, groups=groups)
    
    # --- 5. Save Results ---
    output_dir = derivatives_dir / "mvpa" / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject_id}_target-{target}_decoding-scores.csv"
    
    scores = decoder.cv_scores_[list(decoder.cv_scores_.keys())[0]] if is_categorical else decoder.cv_scores_
    scores_df = pd.DataFrame({'score': scores, 'fold': np.arange(1, len(scores) + 1)})
    scores_df.to_csv(output_path, index=False)
    logging.info(f"Decoding scores saved to {output_path}. Mean score: {np.mean(scores):.3f}")

def main():
    """Main function to run the decoding analysis."""
    parser = argparse.ArgumentParser(description="Run MVPA decoding analysis for a single subject.")
    parser.add_argument("subject_id", help="Subject ID (e.g., 'sub-01').")
    parser.add_argument("derivatives_dir", help="Path to the derivatives directory.")
    parser.add_argument("--target", required=True, help="The target variable to decode from the events file.")
    parser.add_argument("--config", default='config/project_config.yaml', help="Path to the project config file.")
    parser.add_argument("--env", default='hpc', choices=['local', 'hpc'], help="Environment from the config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    
    run_subject_level_decoding(
        subject_id=args.subject_id,
        derivatives_dir=Path(args.derivatives_dir),
        target=args.target,
        params=config['analysis_params']
    )

if __name__ == "__main__":
    main()
