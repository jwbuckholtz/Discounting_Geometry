import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.mvpa.run_decoding_analysis import run_decoding
from nilearn.image import new_img_like
from nilearn.maskers import NiftiMasker
from scripts.mvpa.run_decoding_analysis import run_subject_level_decoding

@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory):
    """
    Creates a synthetic BIDS-like dataset for testing.
    Includes a subject with beta maps and corresponding event files.
    The beta maps will have a simple, known pattern for classification.
    """
    tmp_path = tmp_path_factory.mktemp("data")
    
    # Define paths
    sub_id = "01"
    sub_dir = tmp_path / f"sub-{sub_id}"
    func_dir = sub_dir / "func"
    fprep_dir = tmp_path / "derivatives" / "fmriprep" / f"sub-{sub_id}" / "func"
    lss_dir = tmp_path / "derivatives" / "lss" / f"sub-{sub_id}"
    
    # Create directories
    func_dir.mkdir(parents=True, exist_ok=True)
    fprep_dir.mkdir(parents=True, exist_ok=True)
    lss_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Create a simple brain mask ---
    mask_shape = (10, 10, 10)
    mask_affine = np.eye(4)
    mask_data = np.ones(mask_shape, dtype=np.int8)
    mask_img = nib.Nifti1Image(mask_data, mask_affine)
    mask_path = func_dir / f"sub-{sub_id}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    nib.save(mask_img, mask_path)

    # --- Create synthetic beta maps with a known pattern ---
    n_trials = 20
    conditions = ['high_value', 'low_value'] * (n_trials // 2)
    
    # Create a base image
    base_img_data = np.zeros(mask_shape)
    base_img = nib.Nifti1Image(base_img_data, mask_affine)
    
    for i in range(n_trials):
        beta_data = np.random.randn(*mask_shape) * 0.1 # Background noise
        if conditions[i] == 'high_value':
            # Add a "signal" sphere for high_value trials
            beta_data[3:6, 3:6, 3:6] += 1.0
        else:
            # Add a different "signal" sphere for low_value trials
            beta_data[7:9, 7:9, 7:9] += 1.0
        
        beta_img = new_img_like(base_img, beta_data)
        beta_path = lss_dir / f"sub-{sub_id}_trial-{i+1}_condition-{conditions[i]}.nii.gz"
        nib.save(beta_img, beta_path)

    # --- Create a corresponding events file ---
    events_df = pd.DataFrame({
        'trial_type': conditions,
        'trial_index': np.arange(1, n_trials + 1)
    })
    events_path = func_dir / f"sub-{sub_id}_task-discounting_events.tsv"
    events_df.to_csv(events_path, sep='\t', index=False)
    
    return {
        "bids_dir": tmp_path,
        "derivatives_dir": tmp_path / "derivatives",
        "subject_id": sub_id,
        "mask_path": mask_path,
        "n_trials": n_trials
    }

def test_run_subject_level_decoding_integration(synthetic_dataset):
    """
    Integration test for the main decoding function.
    It runs the decoding on the synthetic dataset and checks if the output
    is created and has the correct shape.
    """
    output_dir = synthetic_dataset["derivatives_dir"] / "mvpa"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the decoding analysis
    run_subject_level_decoding(
        subject_id=synthetic_dataset["subject_id"],
        bids_dir=synthetic_dataset["bids_dir"],
        derivatives_dir=synthetic_dataset["derivatives_dir"],
        mask_path=str(synthetic_dataset["mask_path"]),
        space="MNI152NLin2009cAsym"
    )
    
    # Assert that the output file was created
    expected_output_path = output_dir / f"sub-{synthetic_dataset['subject_id']}_decoding-scores.csv"
    assert expected_output_path.exists()
    
    # Assert that the output file has the correct contents
    scores_df = pd.read_csv(expected_output_path)
    assert 'score' in scores_df.columns
    assert 'fold' in scores_df.columns
    assert len(scores_df) == 5 # Default is 5-fold CV
    
    # Since there's a clear pattern, the mean score should be well above chance (0.5)
    assert scores_df['score'].mean() > 0.8 
