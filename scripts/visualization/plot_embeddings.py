
import argparse
import yaml
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.manifold import TSNE
from nilearn.maskers import NiftiMasker
from nilearn import image
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Tuple

from scripts.utils import find_fmriprep_files, setup_logging

def load_config(config_file: Path) -> Dict[str, Any]:
    """Load the project configuration from a YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_data(derivatives_dir: Path, fmriprep_dir: Path, subject_id: str) -> Tuple[Path, Path, Path]:
    """
    Loads paths to the necessary data files for a subject.
    """
    logging.info(f"Loading data for {subject_id}...")
    
    lss_beta_dir = Path(derivatives_dir) / 'lss_betas' / subject_id
    beta_map_path = lss_beta_dir / f'{subject_id}_lss_beta_maps.nii.gz'
    if not beta_map_path.exists():
        raise FileNotFoundError(f"Beta maps not found at {beta_map_path}")

    behavioral_dir = Path(derivatives_dir) / 'behavioral' / subject_id
    behavioral_data_path = behavioral_dir / f'{subject_id}_discounting_with_sv.tsv'
    if not behavioral_data_path.exists():
        raise FileNotFoundError(f"Behavioral data not found at {behavioral_data_path}")
        
    # Find whole-brain mask from fMRIPrep outputs
    _, brain_mask_path, _ = find_fmriprep_files(fmriprep_dir, subject_id)

    logging.info("Data loaded successfully.")
    return beta_map_path, behavioral_data_path, brain_mask_path

def run_embedding(beta_maps_img: nib.Nifti1Image, brain_mask_img: nib.Nifti1Image, method: str = 'tsne', n_components: int = 2, **kwargs) -> np.ndarray:
    """
    Extracts data from beta maps within a brain mask and computes a
    low-dimensional embedding.

    Parameters
    ----------
    beta_maps_path : str
        Path to the 4D NIfTI file of beta maps.
    brain_mask_path : str
        Path to the brain mask NIfTI file.
    method : str, optional
        The embedding method to use ('tsne' or 'mds'). Defaults to 'tsne'.
    n_components : int, optional
        Number of dimensions for the embedding. Defaults to 2.
    **kwargs : dict
        Additional keyword arguments to pass to the embedding model.

    Returns
    -------
    np.ndarray
        The low-dimensional embedding of the beta maps.
    """
    masker = NiftiMasker(mask_img=brain_mask_img, standardize=True)
    masked_data = masker.fit_transform(beta_maps_img)

    logging.info(f"Running {method.upper()} with {n_components} components...")
    
    if method == 'tsne':
        # Default perplexity is often a good starting point, but can be tuned.
        # It should be less than the number of samples.
        perplexity = min(30, masked_data.shape[0] - 1)
        model = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, **kwargs)
    elif method == 'mds':
        # Note: MDS can be very slow on large datasets
        from sklearn.manifold import MDS
        model = MDS(n_components=n_components, random_state=42, **kwargs)
    else:
        raise ValueError(f"Unknown embedding method: {method}")

    embedding = model.fit_transform(masked_data)
    return embedding

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot low-dimensional embeddings of beta maps.")
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-s061)')
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to the project config file')
    parser.add_argument('--env', type=str, default='local', choices=['local', 'hpc'], help='Environment (local or hpc)')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'mds'], help='Embedding method to use')
    parser.add_argument('--color-by', type=str, default='choice', help='Behavioral variable to color the plot by')
    parser.add_argument('--roi-path', type=str, default=None, help='Optional path to a specific ROI mask file. Defaults to the whole-brain mask.')

    args = parser.parse_args()
    
    setup_logging()
    
    config = load_config(args.config)
    paths = config[args.env]
    derivatives_dir = Path(paths['derivatives_dir'])
    fmriprep_dir = Path(paths['fmriprep_dir'])

    # Load data
    beta_maps_path, behavioral_data_path, brain_mask_path = load_data(derivatives_dir, fmriprep_dir, args.subject)
    
    # If a specific ROI is provided, use it instead of the whole-brain mask
    if args.roi_path:
        brain_mask_path = Path(args.roi_path)
        logging.info(f"Using provided ROI mask: {brain_mask_path}")
        if not brain_mask_path.exists():
            raise FileNotFoundError(f"ROI mask not found at {brain_mask_path}")

    beta_maps_img = nib.load(beta_maps_path)
    behavioral_data = pd.read_csv(behavioral_data_path, sep='\t')
    brain_mask_img = nib.load(brain_mask_path)
    
    # Validate the --color-by variable
    if args.color_by not in behavioral_data.columns:
        raise ValueError(f"Coloring variable '{args.color_by}' not found in behavioral data columns: {behavioral_data.columns.tolist()}")

    # Create a mask for valid (non-NaN) trials for the coloring variable
    valid_trials_mask = behavioral_data[args.color_by].notna().values
    
    # Filter beta maps and behavioral data
    logging.info(f"Found {valid_trials_mask.sum()} valid trials out of {len(valid_trials_mask)} for '{args.color_by}'.")
    filtered_betas_img = image.index_img(beta_maps_img, np.where(valid_trials_mask)[0])
    filtered_behavioral_data = behavioral_data[valid_trials_mask].reset_index(drop=True)

    # Run embedding
    embedding = run_embedding(filtered_betas_img, brain_mask_img, method=args.method)

    # Prepare data for plotting
    plot_df = pd.DataFrame(embedding, columns=['Dim1', 'Dim2'])
    # Use the correctly filtered behavioral data for coloring
    plot_df[args.color_by] = filtered_behavioral_data[args.color_by]

    # Plotting
    roi_name = brain_mask_path.stem.replace('_mask', '').replace('.nii', '')
    output_filename = f'{args.subject}_{args.method}_embedding_{roi_name}_by_{args.color_by}.png'
    
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=plot_df,
        x='Dim1',
        y='Dim2',
        hue=args.color_by,
        palette='viridis', 
        s=100,
        alpha=0.8
    )
    plt.title(f'{args.method.upper()} Embedding for {args.subject} in {roi_name}\n(Colored by {args.color_by})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title=args.color_by)
    plt.tight_layout()

    # Save the figure
    output_dir = derivatives_dir / 'visualization' / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    plt.savefig(output_path)
    
    logging.info(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()
