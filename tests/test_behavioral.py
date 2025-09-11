import pytest
import pandas as pd
import numpy as np
import sys
import os
import shutil
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.behavioral.calculate_discount_rates import (
    hyperbolic_discount,
    choice_probability,
    fit_discount_rate,
    _validate_and_prepare_choice_data,
    process_subject_data
)

# --- Fixture for creating a mock BIDS dataset ---

@pytest.fixture
def mock_bids_env(tmp_path):
    """Creates a mock BIDS directory structure for testing file processing."""
    bids_root = tmp_path / "bids_root"
    derivatives_dir = tmp_path / "derivatives"
    onsets_dir = derivatives_dir / "behavioral" / "onsets"
    
    bids_root.mkdir()
    derivatives_dir.mkdir()
    onsets_dir.mkdir(parents=True)

    # --- Create mock event files for various test cases ---
    # Case 1: Standard files for sub-01 with all required columns
    (onsets_dir / "sub-01_task-discountFix_run-01_events.tsv").write_text(
        "onset\tduration\tsmall_amount\tlarge_amount\tlater_delay\tchoice\n1\t1\t10\t20\t30\t1")
    (onsets_dir / "sub-01_task-discountFix_run-02_events.tsv").write_text(
        "onset\tduration\tsmall_amount\tlarge_amount\tlater_delay\tchoice\n1\t1\t15\t25\t35\t0")
    
    # Case 2: Subject ID that is a prefix of another (sub-01 vs sub-010)
    (onsets_dir / "sub-010_task-discountFix_run-01_events.tsv").write_text("onset\tduration\tchoice\n1\t1\t1")

    # Case 3: A file for a different task that should be ignored
    (onsets_dir / "sub-01_task-other_run-01_events.tsv").write_text("onset\tduration\tchoice\n1\t1\t1")

    # Case 4: A file with no run identifier
    (onsets_dir / "sub-02_task-discountFix_events.tsv").write_text(
        "onset\tduration\tsmall_amount\tlarge_amount\tlater_delay\tchoice\n1\t1\t10\t20\t30\tlarger_later")
    
    # Case 5: Duplicate run number for sub-03 (should raise an error)
    (onsets_dir / "sub-03_task-discountFix_run-01_events.tsv").write_text(
        "onset\tduration\tsmall_amount\tlarge_amount\tlater_delay\tchoice\n1\t1\t10\t20\t30\t1")
    (onsets_dir / "sub-03_ses-test_task-discountFix_run-01_events.tsv").write_text(
        "onset\tduration\tsmall_amount\tlarge_amount\tlater_delay\tchoice\n1\t1\t15\t25\t35\t0")

    # Case 6: Malformed data for testing data integrity
    (onsets_dir / "sub-04_task-discountFix_run-01_events.tsv").write_text(
        "small_amount\tlarge_amount\tlater_delay\tchoice\n"
        "10\t20\t30\t1\n"
        "10\t20\t\t0\n"  # Missing value
    )
    (onsets_dir / "sub-05_task-discountFix_run-01_events.tsv").write_text(
        "small_amount\tlarge_amount\tlater_delay\tchoice\n"
        "10\t\t\t\n" # All required values missing
    )
    
    # Case 7: Mixed and tricky choice formats
    (onsets_dir / "sub-06_task-discountFix_run-01_events.tsv").write_text(
        "choice\n"
        "1.0\n"
        "'0.0'\n"
        "larger_later\n"
        "smaller_sooner\n"
        "1\n"
        "0"
    )
    # Case 8: Invalid choice format
    (onsets_dir / "sub-07_task-discountFix_run-01_events.tsv").write_text("choice\ninvalid_choice")


    return {"onsets_dir": onsets_dir, "derivatives_dir": derivatives_dir}


# --- Unit Tests ---

def test_hyperbolic_discount():
    """Unit test for the hyperbolic discount function."""
    # Test with k=0 (no discounting)
    assert hyperbolic_discount(10, 100, 0) == 100
    # Test with a known value
    assert np.isclose(hyperbolic_discount(30, 100, 0.1), 25.0)

def test_choice_probability():
    """Unit test for the softmax choice probability function."""
    # When SVs are equal, probability should be 0.5
    # Let S=50, L=100, D=30, k=1/30 -> SV_later = 100 / (1 + 1/30 * 30) = 50
    params = (1/30, 1.0) # k, tau
    assert np.isclose(choice_probability(params, 50, 100, 30), 0.5)
    
    # When SV_later is much larger, probability should approach 1
    params = (0.01, 0.1) # k, tau
    assert choice_probability(params, 20, 100, 10) > 0.99

def test_fit_discount_rate_parameter_recovery():
    """
    Integration test for the fitting function using parameter recovery.
    We generate synthetic data with known parameters and check if the
    fitting function can recover them.
    """
    # 1. Define ground truth parameters
    true_k = 0.05
    true_tau = 0.8
    n_trials = 200
    
    # 2. Generate synthetic data based on the true parameters
    np.random.seed(42)
    small_amounts = np.random.uniform(10, 40, n_trials)
    large_amounts = np.random.uniform(50, 100, n_trials)
    delays = np.random.randint(5, 100, n_trials)
    
    probs = choice_probability((true_k, true_tau), small_amounts, large_amounts, delays)
    choices_numeric = np.random.binomial(1, probs)
    choices_str = np.where(choices_numeric == 1, 'larger_later', 'smaller_sooner')

    synthetic_df = pd.DataFrame({
        'small_amount': small_amounts,
        'large_amount': large_amounts,
        'later_delay': delays,
        'choice': choices_str
    })
    
    # 3. Fit the model to the synthetic data
    # Create mock parameters required by the function's new API
    mock_params = {
        'initial_params': [0.1, 1.0],  # k, tau
        'k_bounds': [1e-6, 1.0],
        'tau_bounds': [1e-6, 5.0]
    }
    fit_results, cleaned_data = fit_discount_rate(synthetic_df, mock_params)
    
    # 4. Assert that the recovered parameters are close to the true parameters
    assert 'k' in fit_results
    assert 'tau' in fit_results
    assert np.isclose(fit_results['k'], true_k, atol=0.01) # Absolute tolerance of 0.01
    assert np.isclose(fit_results['tau'], true_tau, atol=0.1) # Absolute tolerance of 0.1

# --- New Tests for Data Handling and File Processing ---

def test_process_subject_data_file_finding(mock_bids_env):
    """
    Tests that the script correctly finds event files, ignoring those from other
    subjects or tasks.
    """
    # This test requires a full run of process_subject_data, so we need mock params.
    # We expect it to fail on fitting (no required columns), but we can check the output file.
    params = {'initial_params': [0.1, 1], 'k_bounds': [0, 1], 'tau_bounds': [0, 1]}
    
    # Test for sub-01: should find exactly 2 files and not mix with sub-010
    process_subject_data("sub-01", mock_bids_env['onsets_dir'], mock_bids_env['derivatives_dir'], params)
    output_file = mock_bids_env['derivatives_dir'] / "behavioral" / "sub-01" / "sub-01_discounting_with_sv.tsv"
    assert output_file.exists()
    data = pd.read_csv(output_file, sep='\t')
    # Should have concatenated data from run 1 and run 2 for sub-01 ONLY
    assert len(data) == 2 
    assert all(data['run'].isin([1, 2]))

def test_process_subject_data_run_handling(mock_bids_env):
    """
    Tests correct parsing of run numbers and handling of duplicates and missing run IDs.
    """
    params = {'initial_params': [0.1, 1], 'k_bounds': [0, 1], 'tau_bounds': [0, 1]}
    
    # Test for sub-02: has one file with no run ID, should be assigned run 1.
    process_subject_data("sub-02", mock_bids_env['onsets_dir'], mock_bids_env['derivatives_dir'], params)
    output_file = mock_bids_env['derivatives_dir'] / "behavioral" / "sub-02" / "sub-02_discounting_with_sv.tsv"
    assert output_file.exists()
    data = pd.read_csv(output_file, sep='\t')
    assert data['run'].iloc[0] == 1

    # Test for sub-03: has duplicate run numbers, should raise a ValueError.
    with pytest.raises(ValueError, match="Duplicate.*run"):
        process_subject_data("sub-03", mock_bids_env['onsets_dir'], mock_bids_env['derivatives_dir'], params)

def test_validate_and_prepare_choice_data():
    """Unit tests for the choice column validation and parsing logic."""
    
    # Test valid mixed data types
    df_valid = pd.DataFrame({'choice': ["1.0", "0.0", "larger_later", 1, 0]})
    result = _validate_and_prepare_choice_data(df_valid.copy())
    assert result['choice'].tolist() == [1, 0, 1, 1, 0]
    
    # Test invalid string
    df_invalid = pd.DataFrame({'choice': ["larger_later", "invalid"]})
    with pytest.raises(ValueError, match="Unrecognized choice values"):
        _validate_and_prepare_choice_data(df_invalid)
        
    # Test missing column
    df_missing = pd.DataFrame({'other_col': [1, 2]})
    with pytest.raises(ValueError, match="required column 'choice'"):
        _validate_and_prepare_choice_data(df_missing)

def test_fit_discount_rate_data_integrity():
    """
    Tests that fit_discount_rate handles missing data and avoids mutating the
    original DataFrame.
    """
    params = {'initial_params': [0.1, 1], 'k_bounds': [0, 1], 'tau_bounds': [0, 1]}
    
    # Test 1: Data with NaNs should be handled gracefully (rows dropped)
    df_with_nans = pd.DataFrame({
        'small_amount': [10, 20, 30],
        'large_amount': [20, 30, np.nan],
        'later_delay': [15, 25, 35],
        'choice': [1, 0, 1]
    })
    original_df = df_with_nans.copy()
    
    fit_results, cleaned_data = fit_discount_rate(df_with_nans, params)
    assert 'k' in fit_results
    assert not np.isnan(fit_results['k']) # Should succeed with the valid rows
    
    # Assert that the original DataFrame was not modified
    pd.testing.assert_frame_equal(df_with_nans, original_df)
    
    # Test 2: Data that becomes empty after dropping NaNs
    df_all_nans = pd.DataFrame({
        'small_amount': [np.nan, 10],
        'large_amount': [20, np.nan],
        'later_delay': [15, 25],
        'choice': [1, 0]
    })
    fit_results_empty, cleaned_data_empty = fit_discount_rate(df_all_nans, params)
    assert np.isnan(fit_results_empty['k']) # Should abort and return NaN
