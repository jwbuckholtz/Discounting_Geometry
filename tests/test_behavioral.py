import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.behavioral.calculate_discount_rates import (
    hyperbolic_discount,
    choice_probability,
    fit_discount_rate
)

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
    choices_str = np.where(choices_numeric == 1, 'larger_later', 'sooner_smaller')

    synthetic_df = pd.DataFrame({
        'small_amount': small_amounts,
        'large_amount': large_amounts,
        'later_delay': delays,
        'choice': choices_str
    })
    
    # 3. Fit the model to the synthetic data
    fit_results = fit_discount_rate(synthetic_df)
    
    # 4. Assert that the recovered parameters are close to the true parameters
    assert 'k' in fit_results
    assert 'tau' in fit_results
    assert np.isclose(fit_results['k'], true_k, atol=0.01) # Absolute tolerance of 0.01
    assert np.isclose(fit_results['tau'], true_tau, atol=0.1) # Absolute tolerance of 0.1
