import pytest
import os
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from nilearn import image

# Correctly import the refactored function
from scripts.modeling.run_lss_model import run_lss_for_subject
from scripts.modeling.run_standard_glm import run_standard_glm_for_subject, prepare_run_events, _validate_modulators_across_runs

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
        'zero_modulator': [2, 2, 2, 2],     # Should be dropped
        'run': [1, 1, 1, 1]                 # Add run column for validation
    })
    
    all_modulators = ['sv_modulator', 'zero_modulator', 'missing_modulator']
    
    # --- 2. Validate modulators first (new approach) ---
    valid_modulators = _validate_modulators_across_runs(events_df, all_modulators, [1])
    
    # --- 3. Run the function to be tested ---
    prepared_df = prepare_run_events(events_df.copy(), valid_modulators)
    
    # --- 3. Assertions ---
    # The DataFrame should now be in long format
    
    # Check the main 'mean' regressor
    mean_events = prepared_df[prepared_df['trial_type'] == 'mean']
    assert len(mean_events) == 4
    assert np.allclose(mean_events['modulation'], 1)

    # Check the 'sv_modulator' parametric modulator (should be included)
    if 'sv_modulator' in valid_modulators:
        sv_events = prepared_df[prepared_df['trial_type'] == 'sv_modulator']
        assert len(sv_events) == 4
        assert np.isclose(sv_events['modulation'].mean(), 0.0), \
            "Modulator with variance was not correctly mean-centered."

    # Assert that the zero-variance and missing modulators were excluded during validation
    assert 'zero_modulator' not in valid_modulators, "Zero-variance modulator should have been excluded during validation"
    assert 'missing_modulator' not in valid_modulators, "Missing modulator should have been excluded during validation"
    
    # Assert that only valid modulators appear in the prepared events
    assert 'zero_modulator' not in prepared_df['trial_type'].unique()
    assert 'missing_modulator' not in prepared_df['trial_type'].unique()
        
    # Assert that the 'trial_type' column was added correctly and contains expected values
    assert 'trial_type' in prepared_df.columns
    assert 'mean' in prepared_df['trial_type'].unique()
    assert 'sv_modulator' in prepared_df['trial_type'].unique()
        
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
    # Create realistic choice-dependent value regressors
    choice_values = np.random.choice([0, 1], n_trials)
    svchosen_values = np.random.rand(n_trials) * 10
    svunchosen_values = np.random.rand(n_trials) * 8  # Different range for unchosen
    svdiff_values = svchosen_values - svunchosen_values  # Realistic difference
    large_amount_values = np.random.rand(n_trials) * 20  # Large amount regressor
    
    events_df = pd.DataFrame({
        'onset': np.linspace(5, (n_scans - 10) * tr, n_trials),
        'duration': np.ones(n_trials) * 1.0,
        'trial_type': ['trial'] * n_trials,  # Add trial_type column for LSS script
        'choice': choice_values,
        'SVchosen': svchosen_values,
        'SVunchosen': svunchosen_values,
        'SVdiff': svdiff_values,
        'large_amount': large_amount_values,
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
        'run_numbers': [1],
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
                'model_specifications': {
                    'choice': ['choice'],
                    'value_chosen': ['SVchosen'],
                    'value_unchosen': ['SVunchosen'], 
                    'value_difference': ['SVdiff'],
                    'large_amount': ['large_amount']
                }
            },
            'run_start_times': {
                '1': 0  # Explicit run timing for test
            }
        }
    }
    
    return {
        "subject_data": subject_data,
        "params": params
    }

def test_run_lss_for_subject_integration(synthetic_glm_dataset):
    # This is a basic smoke test for now
    data = synthetic_glm_dataset["subject_data"]
    params = synthetic_glm_dataset["params"]
    run_lss_for_subject(data, params)

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
    
    # --- Assertions for New Multi-Model Structure ---
    base_output_dir = data["derivatives_dir"] / 'standard_glm' / data["subject_id"]
    
    # Check for choice model outputs
    choice_model_dir = base_output_dir / 'model-choice'
    if choice_model_dir.exists():
        choice_mean_contrast = choice_model_dir / 'contrast-mean_zmap.nii.gz'
        choice_regressor_contrast = choice_model_dir / 'contrast-choice_zmap.nii.gz'
        assert choice_mean_contrast.exists(), "Choice model 'mean' contrast map was not created."
        
    # Check for SVchosen model outputs  
    svchosen_model_dir = base_output_dir / 'model-value_chosen'
    if svchosen_model_dir.exists():
        svchosen_mean_contrast = svchosen_model_dir / 'contrast-mean_zmap.nii.gz'
        svchosen_regressor_contrast = svchosen_model_dir / 'contrast-SVchosen_zmap.nii.gz'
        assert svchosen_mean_contrast.exists(), "SVchosen model 'mean' contrast map was not created."
        assert svchosen_regressor_contrast.exists(), "SVchosen regressor contrast map was not created."
    
    # Verify that at least one model ran successfully
    successful_models = []
    for model_name in ['choice', 'value_chosen', 'value_unchosen', 'value_difference', 'large_amount']:
        model_dir = base_output_dir / f'model-{model_name}'
        if model_dir.exists():
            successful_models.append(model_name)
    
    assert len(successful_models) > 0, f"No GLM models ran successfully. Expected at least one model to complete."
