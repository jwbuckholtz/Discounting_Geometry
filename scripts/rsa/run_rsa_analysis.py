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
from sklearn.model_selection import KFold
from pathlib import Path
from sklearn.base import BaseEstimator

from scripts.utils import load_config, find_fmriprep_files, find_lss_beta_maps, find_behavioral_sv_file

def resample_roi_to_betas(roi_img, beta_maps_img):
    """Resamples an ROI mask to match the space of the beta maps."""
    print("  - Resampling ROI to match beta map space...")
    return image.resample_to_img(roi_img, beta_maps_img, interpolation='nearest')

def load_data(subject_id, derivatives_dir, fmriprep_dir):
    """
    Loads the necessary data for a single subject's RSA.
    """
    # 1. Find and load the single-trial beta maps
    betas_path = find_lss_beta_maps(derivatives_dir, subject_id)
    beta_maps_img = image.load_img(betas_path)

    # 2. Find and load the processed behavioral data
    events_path = find_behavioral_sv_file(derivatives_dir, subject_id)
    events_df = pd.read_csv(events_path, sep='\t')
    
    # 3. Find and load the whole-brain mask
    _, mask_path, _ = find_fmriprep_files(fmriprep_dir, subject_id)
    mask_img = image.load_img(mask_path)

    return beta_maps_img, events_df, mask_img

def create_neural_rdm(beta_maps_img, mask_img):
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
    
    print(f"Created neural RDM with shape: {neural_rdm.shape}")
    
    return neural_rdm

def create_theoretical_rdm(events_df, variable, valid_trials_mask):
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
        print(f"Created categorical RDM for '{variable}'")
    else:
        # --- Continuous RDM (e.g., for SV measures) ---
        # Calculate the absolute difference between each pair of values
        values = variable_data.values
        rdm = np.abs(values[:, np.newaxis] - values[np.newaxis, :])
        print(f"Created continuous RDM for '{variable}'")
        
    return rdm


def run_rsa(neural_rdm, theoretical_rdms):
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
            print(f"  - RSA Result for '{name}': rho={corr:.3f}")
        else:
            rsa_results[name] = np.nan
            
    return rsa_results

def run_crossval_rsa(voxel_data, theoretical_rdms):
    """
    Performs a cross-validated RSA.

    Args:
        voxel_data (np.array): The masked voxel data (n_trials, n_voxels).
        theoretical_rdms (dict): A dictionary of named theoretical RDMs.

    Returns:
        dict: A dictionary where keys are model names and values are arrays
              of cross-validated Spearman correlation scores.
    """
    n_trials = voxel_data.shape[0]
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    results = {name: [] for name in theoretical_rdms.keys()}

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_trials))):
        print(f"  - Running Fold {fold+1}/10...")
        
        # 1. Calculate the cross-fold neural RDM portion
        neural_rdm_part = cdist(voxel_data[train_idx], voxel_data[test_idx], metric='correlation').flatten()
        
        # 2. Correlate with each theoretical model's cross-fold RDM portion
        for name, rdm in theoretical_rdms.items():
            if rdm is not None:
                theoretical_rdm_part = rdm[train_idx, :][:, test_idx].flatten()
                corr, _ = spearmanr(neural_rdm_part, theoretical_rdm_part)
                results[name].append(corr)

    return {name: np.array(scores) for name, scores in results.items()}


class RSASearchlightEstimator(BaseEstimator):
    """
    A scikit-learn compatible estimator for running RSA within a searchlight.
    """
    def __init__(self, theoretical_rdms_vec):
        self.theoretical_rdms_vec = theoretical_rdms_vec

    def fit(self, X, y=None):
        """
        Fit the RSA model to the data of a single searchlight sphere.

        X : array, shape (n_trials, n_voxels_in_sphere)
            The neural data for the sphere.
        y : None
            Unused, but required for scikit-learn compatibility.
        """
        # 1. Create the neural RDM for the current sphere
        neural_rdm_vec = pdist(X, metric='correlation')
        
        # 2. Correlate with each theoretical RDM
        scores = []
        # Sort keys to ensure the output order is always the same
        for model_name in sorted(self.theoretical_rdms_vec.keys()):
            corr, _ = spearmanr(neural_rdm_vec, self.theoretical_rdms_vec[model_name])
            scores.append(corr)
            
        self.scores_ = np.array(scores)
        return self


def run_searchlight_rsa(beta_maps_img, mask_img, theoretical_rdms):
    """
    Performs a searchlight RSA using a custom scikit-learn estimator.
    """
    # --- 1. Prepare Theoretical RDMs ---
    # Vectorize the theoretical RDMs for efficient correlation.
    theoretical_rdms_vec = {
        name: rdm[np.triu_indices_from(rdm, k=1)]
        for name, rdm in theoretical_rdms.items() if rdm is not None
    }
    
    # --- 2. Instantiate and Run the SearchLight ---
    # The estimator is our custom RSA class.
    estimator = RSASearchlightEstimator(theoretical_rdms_vec=theoretical_rdms_vec)
    
    searchlight = SearchLight(
        mask_img,
        estimator=estimator,
        radius=8,
        n_jobs=-1,
        verbose=10
    )
    
    # The 'y' parameter is not used by our custom estimator, so we can pass None
    # or a dummy variable of the correct length.
    searchlight.fit(beta_maps_img, y=None)
    
    # --- 3. Create and Return Result Maps ---
    model_names = sorted(theoretical_rdms_vec.keys())
    result_maps = {}
    for i, model_name in enumerate(model_names):
        # The searchlight.scores_ attribute is now correctly populated
        # with shape (n_voxels, n_models). We select the column for the current model.
        score_map = searchlight.scores_[:, i]
        
        # Unmask the scores back into a Nifti image
        result_map_img = image.new_img_like(mask_img, score_map.astype('float32'))
        result_maps[model_name] = result_map_img
        
    return result_maps


def save_rdm(subject_id, rdm, model_name, output_dir):
    """Saves a single RDM to a file."""
    rdm_dir = output_dir / 'rdms'
    rdm_dir.mkdir(parents=True, exist_ok=True)
    rdm_path = rdm_dir / f'{subject_id}_model-{model_name}_rdm.tsv'
    pd.DataFrame(rdm).to_csv(rdm_path, sep='\t', header=False, index=False)
    print(f"Saved RDM for model '{model_name}' to {rdm_path}")


def save_searchlight_maps(subject_id, searchlight_results, output_dir):
    """Saves the output maps from a searchlight analysis."""
    maps_dir = output_dir / 'searchlight_maps'
    maps_dir.mkdir(parents=True, exist_ok=True)
    for model_name, map_img in searchlight_results.items():
        map_path = maps_dir / f'{subject_id}_model-{model_name}_searchlight-map.nii.gz'
        map_img.to_filename(map_path)
        print(f"Saved searchlight map for model '{model_name}' to {map_path}")


def save_results(subject_id, rsa_results, output_dir, analysis_name):
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
    
    print(f"Saved RSA results for '{analysis_name}' to {output_path}")


def main():
    """Main function to run the RSA."""
    parser = argparse.ArgumentParser(description="Run Representational Similarity Analysis (RSA).")
    parser.add_argument('--config', default='config/project_config.yaml', help='Path to the project configuration file.')
    parser.add_argument('--env', default='local', choices=['local', 'hpc'], help="Environment from the config file.")
    parser.add_argument('--subject', required=True, help='Subject ID (e.g., sub-s061).')
    parser.add_argument('--analysis-type', default='whole_brain', choices=['whole_brain', 'roi', 'searchlight'],
                        help="The type of RSA to run.")
    parser.add_argument('--roi-path', help="Path to the ROI mask file OR a directory of ROI masks.")
    args = parser.parse_args()
    
    # --- Validation for ROI analysis ---
    if args.analysis_type == 'roi' and not args.roi_path:
        raise ValueError("--roi-path must be provided for 'roi' analysis type.")

    # Load configuration
    config = load_config(args.config)
    env_config = config[args.env]
    derivatives_dir = Path(env_config['derivatives_dir'])
    fmriprep_dir = Path(env_config['fmriprep_dir'])

    # 1. Load data (beta maps and events)
    print(f"Loading data for {args.subject}...")
    beta_maps_img, events_df, mask_img = load_data(args.subject, derivatives_dir, fmriprep_dir)

    # --- Pre-analysis Step: Identify base valid trials (where a choice was made) ---
    base_valid_trials_mask = events_df['choice'].notna().values

    # 2. Create theoretical RDMs (do this once)
    # For each variable, we ensure we only use non-NaN trials
    theoretical_rdms = {}
    for var in ['choice', 'later_delay', 'SVchosen', 'SVunchosen', 'SVsum', 'SVdiff']:
        # Create a specific mask for the current variable, combined with the base mask
        var_mask = events_df[var].notna().values
        combined_mask = base_valid_trials_mask & var_mask
        
        # Pass the specific, combined mask to the RDM creation function
        theoretical_rdms[var] = create_theoretical_rdm(events_df, var, combined_mask)

    # --- Main Analysis Logic ---
    if args.analysis_type in ['whole_brain', 'searchlight']:
        # --- Handle single-mask analyses ---
        output_dir = derivatives_dir / 'rsa' / args.subject
        
        # Filter beta maps to only include base valid trials *before* analysis
        beta_maps_valid = image.index_img(beta_maps_img, base_valid_trials_mask)

        if args.analysis_type == 'searchlight':
            # --- Run Searchlight RSA ---
            print("\n--- Running Searchlight RSA ---")
            searchlight_results = run_searchlight_rsa(beta_maps_valid, mask_img, theoretical_rdms)
            save_searchlight_maps(args.subject, searchlight_results, output_dir)
            print("Searchlight analysis complete.")
        else: # whole_brain
            # --- Run Whole-Brain RSA ---
            print("\n--- Running Whole-Brain RSA ---")
            neural_rdm = create_neural_rdm(beta_maps_valid, mask_img)
            
            # For whole-brain, we will run the cross-validated RSA
            masker = NiftiMasker(mask_img=mask_img, standardize=True)
            voxel_data = masker.fit_transform(beta_maps_valid)
            rsa_results = run_crossval_rsa(voxel_data, theoretical_rdms)
            save_results(args.subject, rsa_results, output_dir, 'whole_brain')
            
    elif args.analysis_type == 'roi':
        # --- Handle ROI analysis (single file or directory) ---
        roi_path = Path(args.roi_path)
        if roi_path.is_file():
            roi_files = [roi_path]
        else:
            roi_files = sorted(list(roi_path.glob('*.nii.gz')) + list(roi_path.glob('*.nii')))
            print(f"Found {len(roi_files)} ROI masks in directory: {roi_path}")
            
        for roi_file in roi_files:
            roi_name = roi_file.stem.replace('_mask', '')
            print(f"\n--- Running Cross-validated RSA for ROI: {roi_name} ---")
            
            # Load the original ROI mask
            original_roi_mask = image.load_img(roi_file)

            # Resample the ROI to match the beta maps' space
            analysis_mask_img = resample_roi_to_betas(original_roi_mask, beta_maps_img)
            
            # Extract voxel data from the ROI
            masker = NiftiMasker(mask_img=analysis_mask_img, standardize=True)
            beta_maps_valid = image.index_img(beta_maps_img, base_valid_trials_mask)
            voxel_data = masker.fit_transform(beta_maps_valid)
            
            # Run the cross-validated RSA
            rsa_results = run_crossval_rsa(voxel_data, theoretical_rdms)
            
            # Save the cross-validated results
            output_dir = derivatives_dir / 'rsa' / args.subject
            save_results(args.subject, rsa_results, output_dir, f'roi-{roi_name}')

    # --- Save Theoretical RDMs (only once) ---
    output_dir = derivatives_dir / 'rsa' / args.subject
    for name, rdm in theoretical_rdms.items():
        if rdm is not None:
            save_rdm(args.subject, rdm, name, output_dir)

    print(f"\nSuccessfully finished RSA for {args.subject}")

if __name__ == '__main__':
    main()
