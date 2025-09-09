import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image
import json
import sys
import yaml

from scripts.rsa.run_rsa_analysis import main as rsa_main

@pytest.fixture(scope="module")
def synthetic_rsa_dataset(tmp_path_factory):
    """Creates a synthetic dataset for testing the RSA script."""
    tmp_dir = tmp_path_factory.mktemp("data")
    
    sub_id = "sub-01"
    derivatives_dir = tmp_dir / "derivatives"
    fmriprep_dir = derivatives_dir / "fmriprep"
    
    # --- 1. Create directory structure ---
    lss_dir = derivatives_dir / "lss_betas" / sub_id
    lss_dir.mkdir(parents=True, exist_ok=True)
    behavioral_dir = derivatives_dir / "behavioral" / sub_id
    behavioral_dir.mkdir(parents=True, exist_ok=True)
    fmriprep_func_dir = fmriprep_dir / sub_id / "ses-1" / "func"
    fmriprep_func_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 2. Create synthetic data parameters ---
    n_trials = 40
    shape = (10, 10, 10)
    affine = np.eye(4)
    
    # --- 3. Create beta maps with a known pattern ---
    # The pattern: trials where choice==1 have activation in one region,
    # and trials where choice==0 have activation in another.
    choices = np.random.choice([0, 1], n_trials)
    beta_maps_data = np.random.randn(shape[0], shape[1], shape[2], n_trials) * 0.1
    
    for i in range(n_trials):
        if choices[i] == 1:
            beta_maps_data[2:4, 2:4, 2:4, i] += 1.0 # Region A
        else:
            beta_maps_data[6:8, 6:8, 6:8, i] += 1.0 # Region B
            
    beta_maps_img = nib.Nifti1Image(beta_maps_data, affine)
    beta_maps_path = lss_dir / f"{sub_id}_lss_beta_maps.nii.gz"
    nib.save(beta_maps_img, beta_maps_path)
    
    # --- 4. Create behavioral data ---
    events_df = pd.DataFrame({
        "trial_type": ["decision"] * n_trials,
        "choice": choices,
        "SVchosen": np.random.rand(n_trials) * 10, # A random variable
        "run": np.repeat(np.arange(1, 5), n_trials / 4) # 4 runs
    })
    events_path = behavioral_dir / f"{sub_id}_discounting_with_sv.tsv"
    events_df.to_csv(events_path, sep="\t", index=False)
    
    # --- 5. Create a brain mask ---
    mask_data = np.ones(shape, dtype=np.int8)
    mask_img = nib.Nifti1Image(mask_data, affine)
    mask_path = fmriprep_func_dir / f"{sub_id}_ses-1_task-discountFix_run-1_space-MNI_desc-brain_mask.nii.gz"
    # Create dummy bold and confounds to satisfy find_fmriprep_files
    bold_path = fmriprep_func_dir / f"{sub_id}_ses-1_task-discountFix_run-1_space-MNI_desc-preproc_bold.nii.gz"
    confounds_path = fmriprep_func_dir / f"{sub_id}_ses-1_task-discountFix_run-1_desc-confounds_timeseries.tsv"
    nib.save(nib.Nifti1Image(np.zeros(shape), affine), bold_path)
    pd.DataFrame().to_csv(confounds_path, sep="\t")
    nib.save(mask_img, mask_path)
    
    # --- 6. Create a mock project configuration file ---
    config_path = tmp_dir / "project_config.yaml"
    config_data = {
        'local': {
            'derivatives_dir': str(derivatives_dir),
            'fmriprep_dir': str(fmriprep_dir)
        },
        'rsa': {
            'theoretical_models': ['choice', 'SVchosen'],
            'cv_folds': 4
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    
    return {
        "subject_id": "sub-01",
        "derivatives_dir": derivatives_dir,
        "fmriprep_dir": fmriprep_dir,
        "config_path": config_path
    }

def test_run_subject_level_rsa_integration(synthetic_rsa_dataset, monkeypatch):
    """
    Integration test for the main RSA function.
    Checks that the output file is created and that the RSA result
    for the 'choice' model is positive and significant, as expected
    from the synthetic data pattern.
    """
    sub_id = synthetic_rsa_dataset["subject_id"]
    derivatives_dir = synthetic_rsa_dataset["derivatives_dir"]
    
    # Use monkeypatch to simulate command-line arguments
    test_args = [
        "run_rsa_analysis.py",
        "--subject", sub_id,
        "--env", "local", # Using local paths from our mock config
        "--config", str(synthetic_rsa_dataset["config_path"]),
        "--analysis-type", "whole_brain"
    ]
    monkeypatch.setattr(sys, 'argv', test_args)
    
    # Run the main function from the script
    rsa_main()
    
    # --- Assertions ---
    # 1. Check if the output file was created
    output_path = derivatives_dir / 'rsa' / sub_id / 'summary_results' / f'{sub_id}_analysis-whole_brain_rsa-results.tsv'
    assert output_path.exists(), "Output TSV file was not created."
    
    # 2. Check the contents of the output file
    results_df = pd.read_csv(output_path, sep='\t')
    assert "model" in results_df.columns
    assert "correlation" in results_df.columns
    assert "subject_id" in results_df.columns
    assert len(results_df) > 0, "Results dataframe is empty."
    
    # 3. Check the RSA results
    # The correlation for 'choice' should be high and positive because we built a pattern.
    # The correlation for 'SVchosen' should be near zero because it was random.
    choice_corr = results_df[results_df['model'] == 'choice']['correlation'].mean()
    sv_corr = results_df[results_df['model'] == 'SVchosen']['correlation'].mean()
    
    assert choice_corr > 0.3, f"Choice correlation ({choice_corr}) is not strongly positive as expected."
    assert abs(sv_corr) < 0.2, f"SVchosen correlation ({sv_corr}) is not near zero as expected."
