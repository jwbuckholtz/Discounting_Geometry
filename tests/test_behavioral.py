import pytest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.behavioral.calculate_discount_rates import hyperbolic_discount, choice_probability, fit_discount_rate

def test_hyperbolic_discount_basic():
    """
    Test the hyperbolic_discount function with a basic, easily calculated example.
    """
    amount = 100.0
    delay = 10.0
    k = 0.1
    # Expected value = 100 / (1 + 0.1 * 10) = 100 / 2 = 50
    expected_sv = 50.0
    assert hyperbolic_discount(delay, amount, k) == pytest.approx(expected_sv)

def test_hyperbolic_discount_no_delay():
    """
    Test that a reward with no delay is not discounted.
    """
    amount = 50.0
    delay = 0.0
    k = 0.5
    # Expected value = 50 / (1 + 0.5 * 0) = 50 / 1 = 50
    expected_sv = 50.0
    assert hyperbolic_discount(delay, amount, k) == pytest.approx(expected_sv)

def test_hyperbolic_discount_high_k():
    """
    Test that a very high discount rate leads to a very low subjective value.
    """
    amount = 100.0
    delay = 5.0
    k = 100.0
    # Expected value = 100 / (1 + 100 * 5) = 100 / 501
    expected_sv = 100.0 / 501.0
    assert hyperbolic_discount(delay, amount, k) == pytest.approx(expected_sv)

def test_hyperbolic_discount_zero_amount():
    """
    Test that a reward of zero has a subjective value of zero.
    """
    amount = 0.0
    delay = 20.0
    k = 0.05
    expected_sv = 0.0
    assert hyperbolic_discount(delay, amount, k) == pytest.approx(expected_sv)

def test_hyperbolic_discount_vectorized():
    """
    Test that the function works correctly with NumPy arrays as inputs.
    """
    amounts = np.array([100.0, 50.0])
    delays = np.array([10.0, 0.0])
    k = 0.1
    # Expected SVs = [100 / (1 + 0.1*10), 50 / (1 + 0.1*0)] = [50.0, 50.0]
    expected_svs = np.array([50.0, 50.0])
    calculated_svs = hyperbolic_discount(delays, amounts, k)
    assert np.allclose(calculated_svs, expected_svs)

def test_choice_probability_equal_sv():
    """
    Test that choice probability is 0.5 when subjective values are equal.
    """
    sv_sooner = 50.0
    sv_later = 50.0
    tau = 0.5
    # Expected probability is 0.5
    expected_prob = 0.5
    assert choice_probability(sv_sooner, sv_later, tau) == pytest.approx(expected_prob)

def test_choice_probability_later_better():
    """
    Test that choice probability is high when the later option is much better.
    """
    sv_sooner = 20.0
    sv_later = 80.0
    tau = 0.1  # Low temperature for more deterministic choice
    # Probability of choosing later should be very high
    prob = choice_probability(sv_sooner, sv_later, tau)
    assert prob > 0.99

def test_choice_probability_high_temp():
    """
    Test that a high temperature leads to more random choices (closer to 0.5).
    """
    sv_sooner = 20.0
    sv_later = 80.0
    tau = 100.0  # High temperature
    # Probability should be closer to 0.5
    prob = choice_probability(sv_sooner, sv_later, tau)
    assert prob == pytest.approx(0.5, abs=0.1)

def test_fit_discount_rate_recovery():
    """
    Test that the fitting function can recover known parameters from synthetic data.
    """
    # 1. Define true parameters to recover
    true_k = 0.05
    true_tau = 0.8

    # 2. Generate synthetic data based on these parameters
    np.random.seed(42)  # for reproducibility
    n_trials = 100
    sooner_amounts = np.full(n_trials, 20.0)
    sooner_delays = np.zeros(n_trials)
    later_amounts = np.random.uniform(25, 100, n_trials)
    later_delays = np.random.randint(5, 180, n_trials)

    # Calculate subjective values and choice probabilities
    sv_sooner = hyperbolic_discount(sooner_delays, sooner_amounts, true_k)
    sv_later = hyperbolic_discount(later_delays, later_amounts, true_k)
    prob_later = choice_probability(sv_sooner, sv_later, true_tau)
    
    # Simulate choices based on probabilities
    choices = np.random.binomial(1, prob_later) # 1 = chose later, 0 = chose sooner

    # 3. Fit the model to the synthetic data
    result = fit_discount_rate(choices, sooner_amounts, sooner_delays, later_amounts, later_delays)
    
    # 4. Assert that the recovered parameters are close to the true ones
    assert result is not None
    recovered_k = result['k']
    recovered_tau = result['tau']
    
    assert recovered_k == pytest.approx(true_k, abs=0.02)
    assert recovered_tau == pytest.approx(true_tau, abs=0.2)
    assert result['pseudo_r2'] > 0.5 # Check for a good model fit
