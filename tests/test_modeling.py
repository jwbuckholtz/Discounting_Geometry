import pytest
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.modeling.run_lss_modeling import run_lss_modeling

@pytest.fixture
def fake_neuroimaging_data(tmp_path):
    """
    Creates a temporary directory with fake neuroimaging data for testing.
    This fixture provides paths to the created files.
    """
    # Create fake data parameters
    shape = (10, 10, 10, 20) # 4D data: 10x10x10 voxels, 20 time points
    affine = np.eye(4)
    n_trials = 5

    # Create and save fake BOLD data
    bold_data = np.random.randn(*shape)
    bold_img = nib.Nifti1Image(bold_data, affine)
    bold_file = tmp_path / "bold.nii.gz"
    nib.save(bold_img, bold_file)

    # Create and save a fake brain mask
    mask_data = np.ones(shape[:3])
    mask_img = nib.Nifti1Image(mask_data, affine)
    mask_file = tmp_path / "mask.nii.gz"
    nib.save(mask_img, mask_file)

    # Create a fake confounds file
    confounds_data = pd.DataFrame({
        'trans_x': np.random.randn(shape[3]),
        'trans_y': np.random.randn(shape[3]),
        'trans_z': np.random.randn(shape[3]),
        'rot_x': np.random.randn(shape[3]),
        'rot_y': np.random.randn(shape[3]),
        'rot_z': np.random.randn(shape[3]),
        'a_comp_cor_00': np.random.randn(shape[3]),
        'a_comp_cor_01': np.random.randn(shape[3]),
        'a_comp_cor_02': np.random.randn(shape[3]),
        'a_comp_cor_03': np.random.randn(shape[3]),
        'a_comp_cor_04': np.random.randn(shape[3]),
    })
    confounds_file = tmp_path / "confounds.tsv"
    confounds_data.to_csv(confounds_file, sep='\t', index=False)

    # Create a fake events DataFrame
    events_df = pd.DataFrame({
        'onset': np.arange(n_trials) * 4,
        'duration': np.ones(n_trials)
    })

    return str(bold_file), str(mask_file), str(confounds_file), events_df, n_trials

def test_run_lss_modeling(fake_neuroimaging_data):
    """
    Integration test for the run_lss_modeling function.
    
    This test checks that the function:
    1. Runs without raising an error.
    2. Produces a valid 4D NIfTI image as output.
    3. The output image has the correct number of volumes (equal to the number of trials).
    """
    # Unpack the test data from the fixture
    bold_file, mask_file, confounds_file, events_df, n_trials = fake_neuroimaging_data

    # Run the LSS modeling function
    beta_maps = run_lss_modeling(
        subject_id='sub-test',
        bold_file=bold_file,
        mask_file=mask_file,
        confounds_file=confounds_file,
        events_df=events_df
    )

    # 1. Check that the output is a Nifti1Image
    assert isinstance(beta_maps, nib.Nifti1Image)

    # 2. Check that the output is 4D
    assert beta_maps.ndim == 4

    # 3. Check that the number of volumes in the output matches the number of trials
    assert beta_maps.shape[3] == n_trials
