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

@pytest.fixture
def fake_decoding_data(tmp_path):
    """
    Creates a temporary directory with fake data for decoding analysis.
    """
    # Create fake data parameters
    shape_3d = (10, 10, 10)
    n_trials = 20
    shape_4d = (*shape_3d, n_trials)
    affine = np.eye(4)

    # Create and save fake beta maps (4D)
    beta_data = np.random.randn(*shape_4d)
    beta_img = nib.Nifti1Image(beta_data, affine)
    beta_file = tmp_path / "betas.nii.gz"
    nib.save(beta_img, beta_file)

    # Create and save a fake brain mask (3D)
    mask_data = np.ones(shape_3d)
    mask_img = nib.Nifti1Image(mask_data, affine)
    mask_file = tmp_path / "mask.nii.gz"
    nib.save(mask_img, mask_file)

    # Create a fake behavioral DataFrame
    labels = pd.DataFrame({
        'trial_index': np.arange(n_trials),
        'choice': np.random.choice([0, 1], size=n_trials), # Categorical
        'SVdiff': np.random.randn(n_trials) * 10 # Continuous
    })

    return str(beta_file), str(mask_file), labels

def test_run_decoding_classification(fake_decoding_data):
    """
    Test the run_decoding function for a classification task.
    """
    betas_path, mask_path, labels = fake_decoding_data
    target_variable = 'choice'
    
    scores = run_decoding(
        betas_path=betas_path,
        labels=labels,
        target_variable=target_variable,
        mask_path=mask_path
    )
    
    # Assertions
    assert isinstance(scores, np.ndarray)
    assert len(scores) > 0
    # For classification, scores should be between 0 and 1 (accuracy)
    assert all(0.0 <= score <= 1.0 for score in scores)

def test_run_decoding_regression(fake_decoding_data):
    """
    Test the run_decoding function for a regression task.
    """
    betas_path, mask_path, labels = fake_decoding_data
    target_variable = 'SVdiff'

    scores = run_decoding(
        betas_path=betas_path,
        labels=labels,
        target_variable=target_variable,
        mask_path=mask_path
    )

    # Assertions
    assert isinstance(scores, np.ndarray)
    assert len(scores) > 0
    # For regression, R^2 scores can be negative, so we just check the type
    assert all(isinstance(score, float) for score in scores)
