import pytest
import os
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import image

# Correctly import the refactored function
from scripts.modeling.run_lss_model import run_lss_for_subject
from scripts.modeling.run_standard_glm import run_standard_glm_for_subject, prepare_run_events

# Helper function to create fake data for testing
def create_fake_nifti(shape=(10, 10, 10, 20), affine=np.eye(4)):
    """Creates a fake Nifti image for testing."""
    data = np.random.randn(*shape)
    return nib.Nifti1Image(data, affine)

def test_prepare_run_events_unit():
    """
    Unit test for the event preparation logic in the standard GLM.
    - Tests mean-centering of parametric modulators.
    - Tests addition of missing modulator columns.
    - Tests handling of zero-variance modulators.
    """
    # --- 1. Create a synthetic events DataFrame ---
    events_df = pd.DataFrame({
        'onset': [10, 20, 30, 40],
        'duration': [1, 1, 1, 1],
        'sv_modulator': [1, 2, 3, 4],       # Should be mean-centered
        'zero_modulator': [2, 2, 2, 2]      # Should remain 2 (mean-centered to 0 later)
    })
    
    all_modulators = ['sv_modulator', 'zero_modulator', 'missing_modulator']
    
    # --- 2. Run the function to be tested ---
    prepared_df = prepare_run_events(events_df.copy(), all_modulators)
    
    # --- 3. Assertions ---
    # Assert that the mean of the 'sv_modulator' is now close to zero
    assert np.isclose(prepared_df['sv_modulator'].mean(), 0.0), \
        "Modulator with variance was not correctly mean-centered."
        
    # Assert that the zero-variance modulator is now all zeros after centering
    assert np.allclose(prepared_df['zero_modulator'], 0.0), \
        "Zero-variance modulator was not correctly centered to zero."
        
    # Assert that the 'missing_modulator' was added and is all zeros
    assert 'missing_modulator' in prepared_df.columns, \
        "Missing modulator column was not added."
    assert np.allclose(prepared_df['missing_modulator'], 0.0), \
        "Missing modulator column was not filled with zeros."
        
    # Assert that the 'trial_type' column was added correctly
    assert 'trial_type' in prepared_df.columns and (prepared_df['trial_type'] == 'mean').all(), \
        "'trial_type' column was not added or has incorrect values."
        
    # Assert that the original columns were not dropped
    assert 'onset' in prepared_df.columns and 'duration' in prepared_df.columns, \
        "Original onset/duration columns were dropped."

@pytest.fixture(scope="module")
def synthetic_glm_dataset(tmp_path_factory):
    """Creates a comprehensive synthetic BIDS dataset for testing GLM models."""
    tmp_dir = tmp_path_factory.mktemp("data")
    
    sub_id = "sub-01"
    derivatives_dir = tmp_dir / "derivatives"
    
    # --- 1. Create directory structure ---
    behavioral_dir = derivatives_dir / "behavioral" / sub_id
    behavioral_dir.mkdir(parents=True, exist_ok=True)
    fmriprep_dir = derivatives_dir / "fmriprep"
    fmriprep_func_dir = fmriprep_dir / sub_id / "ses-1" / "func"
    fmriprep_func_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 2. Create synthetic data parameters ---
    n_trials = 20
    n_scans = 100
    tr = 2.0
    shape = (10, 10, 10)
    affine = np.eye(4)
    
    # --- 3. Create a 4D BOLD image ---
    bold_data = np.random.randn(shape[0], shape[1], shape[2], n_scans)
    bold_img = nib.Nifti1Image(bold_data, affine)
    bold_path = fmriprep_func_dir / f"{sub_id}_ses-1_task-discountFix_run-1_space-MNI_desc-preproc_bold.nii.gz"
    nib.save(bold_img, bold_path)
    
    # --- 4. Create an events file ---
    events_df = pd.DataFrame({
        'onset': np.linspace(5, (n_scans - 10) * tr, n_trials),
        'duration': np.ones(n_trials) * 1.0,
        'trial_type': ['decision'] * n_trials,
        'choice': np.random.choice([0, 1], n_trials),
        'SVchosen': np.random.rand(n_trials) * 10,
        'run': [1] * n_trials
    })
    events_path = behavioral_dir / f"{sub_id}_discounting_with_sv.tsv"
    events_df.to_csv(events_path, sep="\t", index=False)
    
    # --- 5. Create a confounds file ---
    confounds_df = pd.DataFrame(np.random.randn(n_scans, 4), columns=['tx', 'ty', 'tz', 'rx'])
    confounds_path = fmriprep_func_dir / f"{sub_id}_ses-1_task-discountFix_run-1_desc-confounds_timeseries.tsv"
    confounds_df.to_csv(confounds_path, sep='\t', index=False)
    
    # --- 6. Create a brain mask ---
    mask_data = np.ones(shape, dtype=np.int8)
    mask_img = nib.Nifti1Image(mask_data, affine)
    mask_path = fmriprep_func_dir / f"{sub_id}_ses-1_task-discountFix_run-1_space-MNI_desc-brain_mask.nii.gz"
    nib.save(mask_img, mask_path)
    
    subject_data = {
        'subject_id': sub_id,
        'bold_imgs': [str(bold_path)],
        'mask_file': str(mask_path),
        'events_df': events_df,
        'confounds_dfs': [confounds_df],
        'derivatives_dir': derivatives_dir
    }
    
    # --- Create mock analysis parameters ---
    params = {
        'analysis_params': {
            't_r': 2.0,
            'slice_time_ref': 0.5,
            'smoothing_fwhm': 5.0,
            'glm': {
                'hrf_model': 'glover',
                'drift_model': 'cosine',
                'parametric_modulators': ['SVchosen']
            }
        }
    }
    
    return {
        "subject_data": subject_data,
        "params": params
    }

def test_run_lss_for_subject_integration(synthetic_glm_dataset):
    """
    Integration smoke test for the LSS modeling script.
    Checks that the script runs to completion and produces an output
    beta map image with the correct dimensions.
    """
    subject_data, params, n_trials = synthetic_glm_dataset
    
    # Run the LSS analysis
    run_lss_for_subject(subject_data, params)
    
    # --- Assertions ---
    # 1. Check if the output file was created
    sub_id = subject_data['subject_id']
    derivatives_dir = subject_data['derivatives_dir']
    output_path = derivatives_dir / 'lss_betas' / sub_id / f"{sub_id}_lss_beta_maps.nii.gz"
    assert output_path.exists(), "LSS beta maps file was not created."
    
    # 2. Check the dimensions of the output file
    beta_maps_img = image.load_img(output_path)
    assert beta_maps_img.ndim == 4, "Output is not a 4D image."
    assert beta_maps_img.shape[3] == n_trials, \
        f"Expected {n_trials} beta maps, but found {beta_maps_img.shape[3]}."
    
    mask_img = image.load_img(subject_data['mask_file'])
    assert beta_maps_img.shape[:3] == mask_img.shape, \
        "Shape of beta maps does not match the brain mask."

def test_run_standard_glm_for_subject_integration(synthetic_glm_dataset):
    """
    Integration test for the main GLM function.
    Checks that the script runs and creates the expected contrast map outputs.
    """
    data = synthetic_glm_dataset["subject_data"]
    params = synthetic_glm_dataset["params"]
    
    run_standard_glm_for_subject(
        subject_data=data,
        params=params
    )
    
    # --- Assertions ---
    output_dir = data["derivatives_dir"] / 'standard_glm' / data["subject_id"]
    
    # Check for the main effect contrast
    mean_contrast_path = output_dir / 'contrast-mean_zmap.nii.gz'
    assert mean_contrast_path.exists(), "Main 'mean' contrast map was not created."
    
    # Check for the parametric modulator contrast
    sv_contrast_path = output_dir / 'contrast-SVchosen_zmap.nii.gz'
    assert sv_contrast_path.exists(), "Parametric modulator 'SVchosen' contrast map was not created."
