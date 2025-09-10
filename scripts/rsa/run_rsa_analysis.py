import os
import argparse
import yaml
import pandas as pd
import numpy as np
from glob import glob
from nilearn import image
from nilearn.maskers import NiftiMasker
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr
from nilearn.decoding import SearchLight
from sklearn.model_selection import KFold, GroupKFold
from pathlib import Path
from sklearn.base import BaseEstimator
from typing import Dict, Any, Tuple, List
import nibabel as nib
import logging

from scripts.utils import load_config, find_fmriprep_files, find_lss_beta_maps, find_behavioral_sv_file, setup_logging

def resample_roi_to_betas(roi_img: nib.Nifti1Image, beta_maps_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Resamples an ROI mask to match the space of the beta maps."""
    logging.info("  - Resampling ROI to match beta map space...")
    return image.resample_to_img(roi_img, beta_maps_img, interpolation='nearest')

def load_data(subject_id: str, derivatives_dir: Path, fmriprep_dir: Path) -> Tuple[nib.Nifti1Image, pd.DataFrame, nib.Nifti1Image, np.ndarray]:
    """
    Loads all the necessary data for a single subject's RSA.
    """
    # 1. Find and load the single-trial beta maps
    logging.info("  - Loading LSS beta maps...")
    betas_path = find_lss_beta_maps(derivatives_dir, subject_id)
    beta_maps_img = image.load_img(betas_path)

    # 2. Find and load the processed behavioral data
    logging.info("  - Loading behavioral events...")
    events_path = find_behavioral_sv_file(derivatives_dir, subject_id)
    events_df = pd.read_csv(events_path, sep='\t')
    
    # 3. Find and load the whole-brain mask
    logging.info("  - Loading brain mask...")
    _, mask_path, _ = find_fmriprep_files(fmriprep_dir, subject_id)
    mask_img = image.load_img(mask_path)

    # 4. Extract groups (run identifiers) for cross-validation
    if 'run' not in events_df.columns:
        raise ValueError(f"'run' column not found in events file for {subject_id}")
    groups = events_df['run'].values

    return beta_maps_img, events_df, mask_img, groups

def create_neural_rdm(beta_maps_img: nib.Nifti1Image, mask_img: nib.Nifti1Image) -> np.ndarray:
    """
    Creates a neural RDM from the single-trial beta maps.

    Args:
        beta_maps_img: The 4D Nifti image of single-trial beta maps.
        mask_img: The brain mask Nifti image.

    Returns:
        np.array: A square, symmetric neural RDM.
    """
    # Use Nilearn's NiftiMasker to extract the voxel data for each trial
    # This is the correct way to apply a mask to 4D data.
    masker = NiftiMasker(mask_img=mask_img, standardize=True)
    voxel_data = masker.fit_transform(beta_maps_img)
    
    # The result is a (n_trials, n_voxels) array.
    # We will now calculate the dissimilarity between each pair of rows (trials).
    # The metric 'correlation' computes 1 - Pearson correlation.
    neural_rdm = squareform(pdist(voxel_data, metric='correlation'))
    
    logging.info(f"Created neural RDM with shape: {neural_rdm.shape}")
    
    return neural_rdm

def create_theoretical_rdm(events_df: pd.DataFrame, variable: str, valid_trials_mask: np.ndarray) -> np.ndarray:
    """
    Creates a theoretical RDM based on a specific behavioral variable.

    Args:
        events_df (pd.DataFrame): DataFrame with trial-by-trial behavioral data.
        variable (str): The name of the column to use for the RDM.
        valid_trials_mask (np.array): A boolean mask of valid trials.

    Returns:
        np.array: A square, symmetric RDM.
    """
    # --- 1. Filter the Data ---
    variable_data = events_df.loc[valid_trials_mask, variable]
    n_trials = len(variable_data)

    # --- 2. Handle Categorical vs. Continuous Data ---
    if variable_data.dtype == 'object':
        # --- Categorical RDM (e.g., for 'choice') ---
        # Convert labels to numerical format
        labels = pd.factorize(variable_data)[0]
        # pdist with 'hamming' computes the proportion of disagreeing elements
        # For binary data, this is 0 if same, 1 if different.
        rdm = squareform(pdist(labels[:, np.newaxis], metric='hamming'))
        logging.info(f"Created categorical RDM for '{variable}'")
    else:
        # --- Continuous RDM (e.g., for SV measures) ---
        # Calculate the absolute difference between each pair of values
        values = variable_data.values
        rdm = np.abs(values[:, np.newaxis] - values[np.newaxis, :])
        logging.info(f"Created continuous RDM for '{variable}'")
        
    return rdm


def run_rsa(neural_rdm: np.ndarray, theoretical_rdms: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Correlates the neural RDM with one or more theoretical RDMs.

    Args:
        neural_rdm (np.array): The neural RDM.
        theoretical_rdms (dict): A dictionary of named theoretical RDMs.

    Returns:
        dict: A dictionary of correlation results (Spearman's rho).
    """
    # --- 1. Vectorize the RDMs ---
    # We only use the upper triangle of the matrices to avoid redundant values
    neural_rdm_vec = neural_rdm[np.triu_indices_from(neural_rdm, k=1)]
    
    rsa_results = {}
    
    # --- 2. Correlate with each Theoretical RDM ---
    for name, rdm in theoretical_rdms.items():
        if rdm is not None:
            theoretical_rdm_vec = rdm[np.triu_indices_from(rdm, k=1)]
            
            # --- 3. Calculate Spearman's Correlation ---
            corr, _ = spearmanr(neural_rdm_vec, theoretical_rdm_vec)
            rsa_results[name] = corr
            logging.info(f"  - RSA Result for '{name}': rho={corr:.3f}")
        else:
            rsa_results[name] = np.nan
            
    return rsa_results

def run_crossval_rsa(voxel_data: np.ndarray, theoretical_rdms: Dict[str, np.ndarray], groups: np.ndarray,
cv_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Performs a cross-validated RSA, respecting data groups (e.g., runs).
    """
    n_trials = voxel_data.shape[0]
    # Folds can't exceed number of groups
    n_splits = min(cv_params['rsa']['cv_folds'], len(np.unique(groups)))
    
    cv = GroupKFold(n_splits=n_splits)
    
    results = {name: [] for name in theoretical_rdms.keys()}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X=np.arange(n_trials), groups=groups)):
        logging.info(f"  - Running Fold {fold+1}/{cv.n_splits}...")
        
        # 1. Calculate the cross-fold neural RDM portion
        neural_rdm_part = cdist(voxel_data[train_idx], voxel_data[test_idx], metric='correlation').flatten()
        
        # 2. Correlate with each theoretical model's cross-fold RDM portion
        for name, rdm in theoretical_rdms.items():
            if rdm is not None:
                theoretical_rdm_part = rdm[train_idx, :][:, test_idx].flatten()
                # Ensure no NaNs from empty models
                if np.isnan(theoretical_rdm_part).any() or np.isnan(neural_rdm_part).any():
                    corr = np.nan
                else:
                    corr, _ = spearmanr(neural_rdm_part, theoretical_rdm_part)
                results[name].append(corr)

    return {name: np.array(scores) for name, scores in results.items()}


class RSASearchlightEstimator:
    """
    A scikit-learn compatible estimator for running RSA within a searchlight.
    """
    def __init__(self):
        self.theoretical_rdm_ = None # Initialize to None

    def fit(self, X, y, theoretical_rdm=None, **kwargs):
        """
        Stores the full, square theoretical RDM. 
        The 'y' contains the full set of indices (0, 1, ..., n_trials-1).
        """
        if theoretical_rdm is None:
            raise ValueError("A theoretical_rdm must be provided to the fit method.")
        self.theoretical_rdm_ = theoretical_rdm
        return self

    def score(self, X, y, **kwargs):
        """
        'X' is the neural data for the sphere (n_trials_in_fold x n_voxels).
        'y' contains the ORIGINAL indices of the trials in this fold.
        """
        try:
            # 1. Create neural RDM for the current fold's data
            neural_rdm_flat = pdist(X, metric='correlation')

            # 2. Subset the full theoretical RDM to match this fold
            # Use np.ix_ to select the correct rows and columns from the square RDM
            fold_rdm_square = self.theoretical_rdm_[np.ix_(y, y)]
            fold_rdm_flat = squareform(fold_rdm_square, checks=False)

            # 3. Correlate the two vectorized RDMs
            correlation, _ = spearmanr(neural_rdm_flat, fold_rdm_flat)
        except ValueError:
            return 0.0 # Handle spheres with zero variance

        return correlation if not np.isnan(correlation) else 0.0

    def get_params(self, deep=True):
        return {}

def run_searchlight_rsa(beta_maps_img: nib.Nifti1Image, theoretical_rdm: np.ndarray, groups: np.ndarray, mask_img: nib.Nifti1Image, params: Dict[str, Any]) -> nib.Nifti1Image:
    """Runs a searchlight RSA analysis."""
    
    # Safeguard against asking for more splits than there are groups
    n_groups = len(np.unique(groups))
    # CRITICAL FIX: The params dict is now the 'rsa' sub-dict
    n_splits = params['cv_folds']
    if n_splits > n_groups:
        logging.warning(f"Requested {n_splits} CV splits, but only {n_groups} groups are available. Setting n_splits to {n_groups}.")
        n_splits = n_groups
    
    cv = GroupKFold(n_splits=n_splits)
    estimator = RSASearchlightEstimator()
    
    searchlight = SearchLight(
        mask_img=mask_img,
        estimator=estimator,
        radius=params['searchlight_radius'],
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    # We pass a dummy 'y' variable for scikit-learn compatibility.
    # The actual SQUARE theoretical RDM is passed as a fit_param to the estimator.
    n_trials = beta_maps_img.shape[-1]
    dummy_y = np.arange(n_trials) # These are the indices that will be passed to score()
    
    # The SearchLight passes extra parameters to the estimator's fit method
    searchlight.fit(beta_maps_img, dummy_y, groups=groups, theoretical_rdm=theoretical_rdm)
    
    scores_1d = searchlight.scores_
    
    # Reshape the 1D scores to 2D for inverse_transform
    scores_2d = scores_1d.reshape(-1, 1)
    
    return searchlight.masker_.inverse_transform(scores_2d)


def save_rdm(subject_id: str, rdm: np.ndarray, model_name: str, output_dir: Path) -> None:
    """Saves a single RDM to a file."""
    rdm_dir = output_dir / 'rdms'
    rdm_dir.mkdir(parents=True, exist_ok=True)
    rdm_path = rdm_dir / f'{subject_id}_model-{model_name}_rdm.tsv'
    pd.DataFrame(rdm).to_csv(rdm_path, sep='\t', header=False, index=False)
    logging.info(f"Saved RDM for model '{model_name}' to {rdm_path}")


def save_searchlight_maps(subject_id: str, searchlight_results: Dict[str, nib.Nifti1Image], output_dir: Path) -> None:
    """Saves the output maps from a searchlight analysis."""
    maps_dir = output_dir / 'searchlight_maps'
    maps_dir.mkdir(parents=True, exist_ok=True)
    for model_name, map_img in searchlight_results.items():
        map_path = maps_dir / f'{subject_id}_model-{model_name}_searchlight-map.nii.gz'
        map_img.to_filename(map_path)
        logging.info(f"Saved searchlight map for model '{model_name}' to {map_path}")


def save_results(subject_id: str, rsa_results: Dict[str, Any], output_dir: Path, analysis_name: str) -> None:
    """
    Saves the RSA results to a file.
    
    Args:
        subject_id (str): The subject ID.
        rsa_results (dict): The dictionary of RSA correlation results.
        output_dir (str): The directory to save the results in.
    """
    all_results = []
    for model, result in rsa_results.items():
        if isinstance(result, (list, np.ndarray)):
            # Cross-validated results
            for i, score in enumerate(result):
                all_results.append({'model': model, 'correlation': score, 'fold': i + 1})
        else:
            # Single result
            all_results.append({'model': model, 'correlation': result, 'fold': 'N/A'})

    results_df = pd.DataFrame(all_results)
    results_df['subject_id'] = subject_id
    results_df['analysis'] = analysis_name
    
    results_dir = output_dir / 'summary_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = results_dir / f'{subject_id}_analysis-{analysis_name}_rsa-results.tsv'
    results_df.to_csv(output_path, sep='\t', index=False)
    
    logging.info(f"Saved RSA results for '{analysis_name}' to {output_path}")

def main() -> None:
    """Main function to run the RSA."""
    parser = argparse.ArgumentParser(description="Run Representational Similarity Analysis (RSA).")
    parser.add_argument('--config', default='config/project_config.yaml', help='Path to the project configuration file.')
    parser.add_argument('--env', default='local', choices=['local', 'hpc'], help="Environment from the config file.")
    parser.add_argument('--subject', required=True, help='Subject ID (e.g., sub-s061).')
    parser.add_argument('--analysis-type', default='whole_brain', choices=['whole_brain', 'roi', 'searchlight'],
                        help="The type of RSA to run.")
    parser.add_argument('--roi-path', help="Path to the ROI mask file OR a directory of ROI masks.")
    args = parser.parse_args()
    
    setup_logging()
    
    # --- Validation for ROI analysis ---
    if args.analysis_type == 'roi' and not args.roi_path:
        raise ValueError("--roi-path must be provided for 'roi' analysis type.")

    # Load configuration
    config = load_config(args.config)
    env_config = config[args.env]
    analysis_params = config['analysis_params']
    derivatives_dir = Path(env_config['derivatives_dir'])
    fmriprep_dir = Path(env_config['fmriprep_dir'])

    # 1. Load data (LSS betas, events, and mask)
    logging.info(f"Loading data for {args.subject}...")
    beta_maps_img, events_df, mask_img, groups = load_data(
        subject_id=args.subject,
        derivatives_dir=derivatives_dir,
        fmriprep_dir=fmriprep_dir
    )

    # --- CRITICAL FIX: Normalize onsets to be relative to the start of each run ---
    # The onset times in the behavioral files are cumulative across the session.
    # We must create a new events dataframe where onsets are relative to their run's start time.
    corrected_events_list = []
    for run_number in sorted(events_df['run'].unique()):
        run_events_df = events_df[events_df['run'] == run_number].copy()
        if not run_events_df.empty:
            first_onset_in_run = run_events_df['onset'].min()
            run_events_df['onset'] -= first_onset_in_run
            logging.info(f"Normalizing onsets for run {run_number} by subtracting {first_onset_in_run:.4f}s")
            corrected_events_list.append(run_events_df)
    
    # Overwrite the original events_df with the corrected one
    events_df = pd.concat(corrected_events_list, ignore_index=True)
    

    # --- Pre-analysis Step: Identify base valid trials (where a choice was made) ---
    base_valid_trials_mask = events_df['choice'].notna().values

    # Get the list of models to run from the config
    models_to_run = analysis_params['rsa']['models']
    
    # Store all RDMs and results in these dicts
    all_theoretical_rdms = {}
    all_rsa_results = {}

    # --- Main Analysis Loop: Iterate through each theoretical model ---
    # This ensures that the neural data is filtered with the exact same mask as the theoretical RDM for each model.
    for model_name in models_to_run:
        logging.info(f"\n--- Starting RSA for Theoretical Model: '{model_name}' ---")

        # 1. Create the specific mask for this model's variable
        if model_name not in events_df.columns:
            logging.warning(f"Variable '{model_name}' not in events data. Skipping.")
            continue
        
        var_mask = events_df[model_name].notna().values
        combined_mask = base_valid_trials_mask & var_mask
        
        num_valid_trials = np.sum(combined_mask)
        if num_valid_trials < 20: # A reasonable minimum number of trials
            logging.warning(f"Fewer than 20 valid trials ({num_valid_trials}) for model '{model_name}'. Skipping.")
            continue
        
        logging.info(f"  - Found {num_valid_trials} valid trials for this model.")

        # Pass the specific, combined mask to the RDM creation function
        theoretical_rdm = create_theoretical_rdm(events_df, model_name, combined_mask)
        all_theoretical_rdms[model_name] = theoretical_rdm # Store for saving later
        
        # 3. Filter the neural data and group labels using the *specific* mask for this model
        beta_maps_valid = image.index_img(beta_maps_img, np.where(combined_mask)[0])
        groups_valid = groups[combined_mask]

        # --- Analysis Execution ---
        if args.analysis_type in ['whole_brain', 'searchlight']:
            output_dir = derivatives_dir / 'rsa' / args.subject
            
            if args.analysis_type == 'searchlight':
                logging.info(f"--- Running Searchlight RSA for model: {model_name} ---")
                # Pass the entire rsa params dict to the function
                searchlight_map = run_searchlight_rsa(
                    beta_maps_valid, 
                    theoretical_rdm, 
                    groups_valid, 
                    mask_img,
                    analysis_params['rsa'] # Pass the 'rsa' sub-dictionary
                )
                
                # Save the resulting map
                output_dir = derivatives_dir / 'rsa' / args.subject / 'searchlight_maps'
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f'{args.subject}_model-{model_name}_rsa-searchlight-map.nii.gz'
                nib.save(searchlight_map, output_path)
                logging.info(f"Saved searchlight map to {output_path}")

            else: # whole_brain
                logging.info(f"--- Running Whole-Brain RSA for model: {model_name} ---")
                masker = NiftiMasker(mask_img=mask_img, standardize=False)
                voxel_data = masker.fit_transform(beta_maps_valid)
                rsa_results = run_crossval_rsa(voxel_data, {model_name: theoretical_rdm}, groups_valid, analysis_params)
                all_rsa_results.update(rsa_results) # Add this model's results to the collection

        elif args.analysis_type == 'roi':
            roi_path = Path(args.roi_path)
            roi_files = [roi_path] if roi_path.is_file() else sorted(list(roi_path.glob('*.nii.gz')) + list(roi_path.glob('*.nii')))
            
            for roi_file in roi_files:
                roi_name = roi_file.stem.replace('_mask', '')
                logging.info(f"\n--- Running RSA for ROI: {roi_name} ---")
                
                analysis_mask_img = resample_roi_to_betas(image.load_img(roi_file), beta_maps_img)
                masker = NiftiMasker(mask_img=analysis_mask_img, standardize=False)
                voxel_data = masker.fit_transform(beta_maps_valid)
                
                rsa_results = run_crossval_rsa(voxel_data, {model_name: theoretical_rdm}, groups_valid, analysis_params)
                
                # We need to save per-ROI results, so we can't just update the main dict.
                # Let's save them inside the loop.
                output_dir = derivatives_dir / 'rsa' / args.subject
                save_results(args.subject, rsa_results, output_dir, f'roi-{roi_name}')

    # --- Save Aggregated Results and RDMs ---
    output_dir = derivatives_dir / 'rsa' / args.subject
    
    # Save the whole-brain results if they exist
    if args.analysis_type == 'whole_brain' and all_rsa_results:
        save_results(args.subject, all_rsa_results, output_dir, 'whole_brain')

    for name, rdm in all_theoretical_rdms.items():
        if rdm is not None:
            save_rdm(args.subject, rdm, name, output_dir)

    logging.info(f"\nSuccessfully finished all RSA analyses for {args.subject}")

if __name__ == '__main__':
    main()
