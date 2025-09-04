import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.rsa.run_rsa_analysis import create_neural_rdm, create_theoretical_rdm, run_rsa

@pytest.fixture
def fake_rsa_data(tmp_path):
    """
    Creates a temporary directory with fake data for RSA analysis.
    The neural data is intentionally structured to have a clear pattern.
    """
    # Create fake data parameters
    shape_3d = (5, 5, 5)
    n_trials = 10
    shape_4d = (*shape_3d, n_trials)
    affine = np.eye(4)

    # Create structured beta data: first half of trials are different from the second half
    beta_data = np.zeros(shape_4d)
    beta_data[:, :, :, :n_trials//2] = 1.0  # First 5 trials are all 1s
    beta_data[:, :, :, n_trials//2:] = -1.0 # Last 5 trials are all -1s
    beta_data += np.random.randn(*shape_4d) * 0.1 # Add a small amount of noise

    beta_img = nib.Nifti1Image(beta_data, affine)
    
    # Create a simple brain mask that includes all voxels
    mask_data = np.ones(shape_3d)
    mask_img = nib.Nifti1Image(mask_data, affine)

    # Create a behavioral DataFrame that matches the neural structure
    behavioral_data = pd.DataFrame({
        'trial_index': np.arange(n_trials),
        'condition': ['A'] * (n_trials//2) + ['B'] * (n_trials//2), # Categorical
        'value': np.arange(n_trials) # Continuous (uncorrelated with structure)
    })

    return beta_img, mask_img, behavioral_data

def test_rdm_creation_and_rsa_correlation(fake_rsa_data):
    """
    Integration test for the core RSA functions.
    
    This test checks that:
    1. Neural and Theoretical RDMs are created with the correct shape.
    2. The RSA correlation can recover a known structure in the data.
    """
    beta_maps_img, mask_img, behavioral_data = fake_rsa_data
    n_trials = behavioral_data.shape[0]

    # 1. Test Neural RDM creation
    neural_rdm = create_neural_rdm(beta_maps_img, mask_img)
    assert neural_rdm.shape == (n_trials, n_trials)
    assert np.all(np.diag(neural_rdm) == 0) # Diagonal should be zero

    # 2. Test Theoretical RDM creation for the structured variable
    theoretical_rdm_structured = create_theoretical_rdm(behavioral_data, 'condition')
    assert theoretical_rdm_structured.shape == (n_trials, n_trials)
    
    # Check a value: trials in the same condition should have 0 distance
    assert theoretical_rdm_structured[0, 1] == 0 
    # Check a value: trials in different conditions should have non-zero distance
    assert theoretical_rdm_structured[0, 9] > 0

    # 3. Test Theoretical RDM for the unstructured variable
    theoretical_rdm_unstructured = create_theoretical_rdm(behavioral_data, 'value')
    assert theoretical_rdm_unstructured.shape == (n_trials, n_trials)

    # 4. Test RSA correlation
    # The correlation between the neural RDM and the structured theoretical RDM should be high
    rsa_result_structured = run_rsa(neural_rdm, theoretical_rdm_structured)
    assert 'rho' in rsa_result_structured
    assert rsa_result_structured['rho'] > 0.8

    # The correlation with the unstructured RDM should be low
    rsa_result_unstructured = run_rsa(neural_rdm, theoretical_rdm_unstructured)
    assert 'rho' in rsa_result_unstructured
    assert abs(rsa_result_unstructured['rho']) < 0.5
