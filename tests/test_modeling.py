import pytest
import os
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path

# Correctly import the refactored function
from scripts.modeling.run_lss_model import run_lss_for_subject

# Helper function to create fake data for testing
def create_fake_nifti(shape=(10, 10, 10, 20), affine=np.eye(4)):
    """Creates a fake Nifti image for testing."""
    data = np.random.randn(*shape)
    return nib.Nifti1Image(data, affine)

@pytest.fixture
def subject_data(tmp_path):
    """Creates a temporary directory with all necessary fake data for a subject."""
    sub_dir = tmp_path / "sub-test"
    sub_dir.mkdir()
    
    derivatives_dir = tmp_path / "derivatives"
    derivatives_dir.mkdir()
    
    # Create fake beta maps and save them
    beta_maps_img = create_fake_nifti()
    lss_betas_dir = derivatives_dir / 'lss_betas' / 'sub-test'
    lss_betas_dir.mkdir(parents=True)
    betas_path = lss_betas_dir / "sub-test_lss_beta_maps.nii.gz"
    nib.save(beta_maps_img, betas_path)

    # Create fake mask
    mask_img = create_fake_nifti(shape=(10, 10, 10))

    # Create fake events dataframe
    events_df = pd.DataFrame({
        'onset': np.arange(0, 20 * 2, 2),
        'duration': [1] * 20,
        'trial_type': ['stim'] * 20
    })

    # Create fake confounds
    confounds_df = pd.DataFrame(np.random.rand(20, 4), columns=['c1', 'c2', 'c3', 'c4'])

    return {
        "subject_id": "sub-test",
        "bold_file": create_fake_nifti(), # In-memory, not saved
        "mask_file": mask_img,
        "events_df": events_df,
        "confounds_selected": confounds_df,
        "derivatives_dir": derivatives_dir
    }

def test_run_lss_model_smoke_test(subject_data):
    """
    A simple 'smoke test' to ensure the LSS modeling script runs without crashing.
    It checks if the output file is created.
    """
    run_lss_for_subject(subject_data)
    
    # Check if the output file was created
    expected_output = (
        subject_data["derivatives_dir"]
        / "lss_betas"
        / subject_data["subject_id"]
        / f"{subject_data['subject_id']}_lss_beta_maps.nii.gz"
    )
    assert expected_output.exists(), "The LSS model script did not create the expected output file."
