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
from nilearn import image

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
    
    # Use the LSS beta directory that the script expects
    lss_dir = tmp_path / "derivatives" / "lss_betas" / f"sub-{sub_id}"
    lss_dir.mkdir(parents=True, exist_ok=True)
    
    beta_maps_list = []
    for i in range(n_trials):
        beta_data = np.random.randn(*mask_shape) * 0.1 # Background noise
        if conditions[i] == 'high_value':
            beta_data[3:6, 3:6, 3:6] += 1.0 # Signal region 1
        else:
            beta_data[7:9, 7:9, 7:9] += 1.0 # Signal region 2
        
        beta_img = nib.Nifti1Image(beta_data, mask_affine)
        beta_maps_list.append(beta_img)
    
    # Save as a single 4D file, as expected by the loader
    beta_maps_4d = image.concat_imgs(beta_maps_list)
    beta_map_path = lss_dir / f"sub-{sub_id}_lss_beta_maps.nii.gz"
    nib.save(beta_maps_4d, beta_map_path)

    # --- Create a corresponding events file ---
    # The loader function now expects this specific naming convention
    beh_dir = tmp_path / "derivatives" / "behavioral" / f"sub-{sub_id}"
    beh_dir.mkdir(parents=True, exist_ok=True)
    events_path = beh_dir / f"sub-{sub_id}_discounting_with_sv.tsv"
    events_df = pd.DataFrame({
        'trial_type': conditions,
        'SVchosen': np.random.randn(n_trials), # Add a continuous variable
        'run': np.repeat([1, 2], n_trials // 2)
    })
    events_df.to_csv(events_path, sep='\t', index=False)
    
    # --- Create mock analysis parameters ---
    params = {
        'mvpa': {
            'classification': {
                'target_variables': ['trial_type'],
                'estimator': 'SVC', # Use uppercase to match code
                'scoring': 'accuracy',
                'cv_folds': 2,
                'random_state': 42
            },
            'regression': {
                'target_variables': ['SVDiff'],
                'estimator': 'SVR', # Use uppercase to match code
                'scoring': 'r2',
                'cv_folds': 2
            }
        }
    }
    
    return {
        "derivatives_dir": tmp_path / "derivatives",
        "fmriprep_dir": tmp_path / "derivatives" / "fmriprep",
        "subject_id": sub_id,
        "n_trials": n_trials,
        "params": params
    }

def test_run_subject_level_decoding_integration(synthetic_dataset):
    """
    Integration test for the main decoding function.
    It runs the decoding on the synthetic dataset and checks if the output
    is created and has the correct shape.
    """
    output_dir = synthetic_dataset["derivatives_dir"] / "mvpa" / f"sub-{synthetic_dataset['subject_id']}"
    
    # Create mock analysis parameters, similar to project_config.yaml
    params = synthetic_dataset["params"]
    
    # Run the decoding analysis for the classification target
    run_subject_level_decoding(
        subject_id=f"sub-{synthetic_dataset['subject_id']}",
        derivatives_dir=synthetic_dataset["derivatives_dir"],
        fmriprep_dir=synthetic_dataset["fmriprep_dir"],
        target='trial_type',
        params=params
    )
    
    # --- Assertions ---
    output_dir = synthetic_dataset["derivatives_dir"] / "mvpa" / f"sub-{synthetic_dataset['subject_id']}"
    # The output filename now contains more metadata
    expected_output_path = output_dir / f"sub-{synthetic_dataset['subject_id']}_target-trial_type_roi-whole_brain_estimator-SVC_scoring-accuracy_decoding-scores.tsv"
    assert expected_output_path.exists()
    
    # Assert that the output file has the correct contents
    scores_df = pd.read_csv(expected_output_path, sep='\t')
    assert 'scores' in scores_df.columns # Check for 'scores' (plural)
    assert 'fold' in scores_df.columns
    assert len(scores_df) == 2 # n_splits = 2
    
    # Since there's a clear pattern, the mean score should be well above chance (0.5)
    assert scores_df['scores'].mean() > 0.8 
