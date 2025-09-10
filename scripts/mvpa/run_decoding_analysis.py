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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

def prepare_decoding_data(events_df: pd.DataFrame, target_variable: str, n_betas: int, is_categorical: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares data for decoding by validating trial counts, extracting labels,
    handling NaNs, and creating group (run) labels.
    The decision to treat data as categorical is now passed in.
    """
    # --- 1. Validation ---
    if target_variable not in events_df.columns:
        raise ValueError(f"Target variable '{target_variable}' not found in events DataFrame.")
    
    if len(events_df) != n_betas:
        raise ValueError(f"Mismatch between number of events ({len(events_df)}) and beta maps ({n_betas}).")

    # --- 2. Create Labels and Trial Mask based on NaNs ---
    labels_series = events_df[target_variable].copy()
    
    if labels_series.isnull().any():
        valid_trials_mask = labels_series.notna().values
        labels_valid = labels_series[valid_trials_mask].values
    else:
        valid_trials_mask = np.ones(len(labels_series), dtype=bool)
        labels_valid = labels_series.values

    # Factorize labels for classification if specified
    if is_categorical:
        final_labels, _ = pd.factorize(labels_valid)
    else:
        final_labels = labels_valid.astype(float)
        
    # --- 3. Get Grouping Variable (for CV) ---
    if 'run' in events_df.columns:
        groups = events_df['run'].values
    else:
        # If no run information, we cannot use GroupKFold.
        # We'll pass None and the downstream function should handle it.
        logging.warning("No 'run' column found in events data. Cannot use GroupKFold for cross-validation.")
        groups = None

    return final_labels, valid_trials_mask, groups

def get_estimator(estimator_name: str, is_categorical: bool) -> Pipeline:
    """
    Gets a scikit-learn estimator pipeline by name.
    The lookup is case-insensitive.
    """
    # Define available estimators with lowercase keys for case-insensitive matching
    estimators = {
        'svc': SVC(kernel='linear', class_weight='balanced'),
        'svr': SVR(kernel='linear'),
        'logisticregression': LogisticRegression(penalty='l2', class_weight='balanced', max_iter=1000),
        'ridge': Ridge(),
        'randomforestclassifier': RandomForestClassifier(class_weight='balanced'),
        'randomforestregressor': RandomForestRegressor()
    }
    
    model = estimators.get(estimator_name.lower())
    
    if model is None:
        raise ValueError(f"Unknown estimator '{estimator_name}'. Available options are: {list(estimators.keys())}")
    
    # Basic check to ensure a classifier is used for classification and vice versa
    is_classifier = hasattr(model, 'predict_proba') or isinstance(model, SVC)
    if is_categorical and not is_classifier:
        logging.warning(f"Estimator '{estimator_name}' is not a classifier but the target is categorical.")
    elif not is_categorical and is_classifier:
        logging.warning(f"Estimator '{estimator_name}' is a classifier but the target is not categorical.")

    return make_pipeline(StandardScaler(), model)

def run_decoding(
    beta_maps_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    labels: np.ndarray,
    valid_trials_mask: np.ndarray,
    groups: np.ndarray,
    is_categorical: bool,
    cv_params: Dict[str, Any],
    estimator: str,
    scoring: str
) -> np.ndarray:
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
    masker = NiftiMasker(mask_img=mask_img, standardize=False)
    # This creates the (n_trials, n_voxels) data matrix
    X = masker.fit_transform(beta_maps_img) 
    
    X_valid = X[valid_trials_mask]
    labels_valid = labels[valid_trials_mask]
    
    # Only try to group if groups are available
    groups_valid = groups[valid_trials_mask] if groups is not None else None
    
    # Use NiftiMasker to extract the time series from the ROIs
    # standardize=False is crucial to prevent data leakage across CV folds.
    # Standardization is correctly handled within the scikit-learn pipeline.
    
    # --- 2. Set up the Cross-Validation ---
    if groups_valid is not None:
        n_groups = len(np.unique(groups_valid))
        if cv_params['n_splits'] > n_groups:
            logging.warning(
                f"Requested {cv_params['n_splits']} CV splits, but only {n_groups} groups (runs) are available. "
                f"Setting n_splits to {n_groups}."
            )
            cv_params['n_splits'] = n_groups
        
        if is_categorical:
            cv = StratifiedGroupKFold(**cv_params)
        else:
            cv = GroupKFold(**cv_params)
    else:
        # If no groups, check n_splits against number of samples
        n_samples = len(labels_valid)
        if cv_params['n_splits'] > n_samples:
            logging.warning(
                f"Requested {cv_params['n_splits']} CV splits, but only {n_samples} samples are available. "
                f"Setting n_splits to {n_samples}."
            )
            cv_params['n_splits'] = n_samples

        if is_categorical:
            cv = StratifiedKFold(**cv_params)
        else:
            cv = KFold(**cv_params)

    # --- 3. Set up the ML model ---
    model = get_estimator(estimator, is_categorical)

    # --- 4. Run Cross-Validation ---
    logging.info(f"Running {cv_params['n_splits']}-fold cross-validation with '{estimator}' and '{scoring}' scoring...")
    
    scores = cross_val_score(
        model,
        X=X_valid, # CRITICAL FIX: Use the masked data matrix, not the NIfTI image
        y=labels_valid,
        cv=cv,
        groups=groups_valid,
        scoring=scoring,
        n_jobs=-1 
    )
    
    return scores


def save_results(subject_id: str, target_variable: str, scores: np.ndarray, output_dir: Path, roi_name: str, estimator: str, scoring: str) -> None:
    """
    Saves the decoding results to a file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        'scores': scores,
        'fold': range(1, len(scores) + 1),
        'subject_id': subject_id,
        'roi': roi_name,
        'target_variable': target_variable,
        'scoring': scoring,
        'estimator': estimator
    })
    
    output_filename = f'{subject_id}_target-{target_variable}_roi-{roi_name}_estimator-{estimator}_scoring-{scoring}_decoding-scores.tsv'
    output_path = output_dir / output_filename
    results_df.to_csv(output_path, sep='\t', index=False)
    
    logging.info(f"Saved decoding scores to {output_path}")
    logging.info(f"Mean score for ROI '{roi_name}': {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")


def run_subject_level_decoding(subject_id: str, derivatives_dir: Path, fmriprep_dir: Path, target: str, params: Dict[str, Any]):
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
    # The 'params' dictionary is analysis_params from the config
    if target in params['mvpa']['classification']['target_variables']:
        analysis_params = params['mvpa']['classification']
        is_categorical = True
    elif target in params['mvpa']['regression']['target_variables']:
        analysis_params = params['mvpa']['regression']
        is_categorical = False
    else:
        raise ValueError(f"Target '{target}' not defined in classification or regression config.")

    # --- 3. Prepare Data ---
    labels, valid_trials, groups = prepare_decoding_data(
        events_df, target, n_betas=beta_maps_img.shape[-1], is_categorical=is_categorical
    )
    
    # --- 4. Get CV Parameters ---
    if is_categorical:
        cv_params = {
            'n_splits': analysis_params['cv_folds']
        }
        # StratifiedGroupKFold does not accept random_state without shuffle,
        # and shuffling is not typically done with grouped data.
    else:
        cv_params = {'n_splits': analysis_params['cv_folds']}
        
    # --- 5. Run Decoding ---
    # We use the whole-brain mask for this simplified script
    _, mask_path, _ = find_fmriprep_files(fmriprep_dir, subject_id)
    
    # Resample the mask to match the beta maps
    resampled_mask = resample_roi_to_betas(mask_path, lss_betas_path)
    
    scores = run_decoding(
        beta_maps_img=beta_maps_img,
        mask_img=resampled_mask,
        labels=labels,
        valid_trials_mask=valid_trials,
        groups=groups,
        is_categorical=is_categorical,
        cv_params=cv_params,
        estimator=analysis_params['estimator'],
        scoring=analysis_params['scoring']
    )
    
    # --- 6. Save Results ---
    output_dir = derivatives_dir / "mvpa" / subject_id
    save_results(subject_id, target, scores, output_dir, roi_name='whole_brain', estimator=analysis_params['estimator'], scoring=analysis_params['scoring'])
    
def main():
    """
    Main function to run the decoding analysis from the command line.
    """
    parser = argparse.ArgumentParser(description="Run MVPA decoding analysis for a single subject.")
    parser.add_argument("subject_id", help="Subject ID (e.g., 'sub-01').")
    parser.add_argument("--target", help="Optional: The target variable to decode. If not provided, all targets in the config file will be run.")
    parser.add_argument("--config", default='config/project_config.yaml', help="Path to the project config file.")
    parser.add_argument("--env", default='hpc', choices=['local', 'hpc'], help="Environment from the config file.")
    args = parser.parse_args()

    # Load the full config
    config = load_config(args.config)
    paths = config[args.env]
    derivatives_dir = Path(paths['derivatives_dir'])
    fmriprep_dir = Path(paths['fmriprep_dir'])
    
    # Use the correct config structure: analysis_params -> mvpa
    analysis_params = config['analysis_params']

    # Determine which targets to run from the single 'targets' list
    targets_to_run = [args.target] if args.target else analysis_params['mvpa']['targets']
    
    for target in targets_to_run:
        try:
            run_subject_level_decoding(
                subject_id=args.subject_id,
                derivatives_dir=derivatives_dir,
                fmriprep_dir=fmriprep_dir,
                target=target,
                params=analysis_params # Pass the correct sub-dict
            )
        except Exception as e:
            logging.error(f"Failed to run decoding for target '{target}'. Error: {e}")
            logging.exception(e)

if __name__ == '__main__':
    main()
