import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import argparse
import yaml
from pathlib import Path

from scripts.utils import load_config

def hyperbolic_discount(delay, amount, k):
    """
    Calculates the subjective value of a delayed reward using a hyperbolic model.

    Args:
        delay (float or np.array): Delay to the reward.
        amount (float or np.array): Amount of the reward.
        k (float): Discount rate parameter.

    Returns:
        float or np.array: Subjective value of the reward.
    """
    return amount / (1 + k * delay)

def choice_probability(params, S, L, D):
    """
    Calculates the probability of choosing the larger, later reward using a softmax function.

    Args:
        params (tuple): A tuple containing (k, tau), where k is the discount
                        rate and tau is the temperature parameter.
        S (float): The amount of the smaller, sooner reward.
        L (float): The amount of the larger, later reward.
        D (float): The delay to the larger, later reward.

    Returns:
        float: The probability of choosing the larger, later reward.
    """
    k, tau = params
    sv_sooner = hyperbolic_discount(0, S, k)  # Sooner reward has 0 delay
    sv_later = hyperbolic_discount(D, L, k)
    
    # A numerically stable implementation of the softmax (logistic function)
    prob_later = 1 / (1 + np.exp(-(sv_later - sv_sooner) / tau))
    return prob_later

def fit_discount_rate(data):
    """
    Fits the hyperbolic discounting model to behavioral data to estimate k and tau.

    Args:
        data (pd.DataFrame): DataFrame with behavioral data for a single subject.

    Returns:
        dict: A dictionary containing the fitted parameters and model fit statistics.
    """
    # Define the negative log-likelihood function to minimize
    def neg_log_likelihood(params, S, L, D, choices):
        probs = choice_probability(params, S, L, D)
        # Clamp probabilities to avoid log(0) errors
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        
        log_likelihood = np.sum(choices * np.log(probs) + (1 - choices) * np.log(1 - probs))
        return -log_likelihood

    # Prepare the data for fitting
    S = data['small_amount'].values
    L = data['large_amount'].values
    D = data['later_delay'].values
    choices = (data['choice'] == 'larger_later').astype(int).values

    # Initial guesses for k and tau
    initial_params = [0.01, 1.0]

    # Minimize the negative log-likelihood
    result = minimize(neg_log_likelihood, initial_params, args=(S, L, D, choices),
                      bounds=[(1e-6, None), (1e-6, None)])

    if not result.success:
        print(f"Warning: Optimization failed. Reason: {result.message}")
        return {'k': np.nan, 'tau': np.nan, 'neg_log_likelihood': np.nan, 'pseudo_r2': np.nan}

    # Calculate pseudo-R-squared (McFadden's R-squared)
    ll_fit = -result.fun
    p_null = np.mean(choices)
    ll_null = np.sum(choices * np.log(p_null) + (1 - choices) * np.log(1 - p_null))
    pseudo_r2 = 1 - (ll_fit / ll_null)
    
    fit_results = {
        'k': result.x[0],
        'tau': result.x[1],
        'neg_log_likelihood': result.fun,
        'pseudo_r2': pseudo_r2
    }
    return fit_results

def calculate_subjective_values(data, k):
    """
    Calculates subjective values for chosen and unchosen options.

    Args:
        data (pd.DataFrame): DataFrame with trial-by-trial data.
        k (float): The subject's discount rate.

    Returns:
        pd.DataFrame: The input DataFrame with added columns for subjective values.
    """
    if k is None or np.isnan(k):
        # If k could not be estimated, fill SV columns with NaNs
        data['SVchosen'] = np.nan
        data['SVunchosen'] = np.nan
        data['SVsum'] = np.nan
        data['SVdiff'] = np.nan
        return data

    # Calculate the subjective value of each option
    sv_sooner = hyperbolic_discount(0, data['small_amount'], k)
    sv_later = hyperbolic_discount(data['later_delay'], data['large_amount'], k)

    # Determine chosen and unchosen SV based on the 'choice' column
    is_later_chosen = (data['choice'] == 'larger_later')
    
    data['SVchosen'] = np.where(is_later_chosen, sv_later, sv_sooner)
    data['SVunchosen'] = np.where(is_later_chosen, sv_sooner, sv_later)

    # Calculate sum and difference
    data['SVsum'] = sv_sooner + sv_later
    data['SVdiff'] = sv_later - sv_sooner # Larger-Later minus Smaller-Sooner

    return data

def process_subject_data(subject_id, onsets_dir, derivatives_dir):
    """
    Processes all behavioral data for a single subject, handling multiple runs.
    """
    onset_subject_id = subject_id.replace('sub-', '')
    
    # Find all event files for this subject, which may be split by run
    event_files = sorted(list(onsets_dir.glob(f'{onset_subject_id}*discountFix_events.tsv')))
    
    if not event_files:
        print(f"Warning: No event files found for {subject_id}")
        return None
        
    # Load and concatenate data from all runs
    all_runs_data = []
    for event_file in event_files:
        # Extract run number from filename, default to 1 if not found
        run_part = event_file.stem.split('_run-')
        run = int(run_part[1].split('_')[0]) if len(run_part) > 1 else 1
        
        run_data = pd.read_csv(event_file, sep='\t')
        run_data['run'] = run
        all_runs_data.append(run_data)
    
    data = pd.concat(all_runs_data, ignore_index=True)

    # Fit the discount rate and temperature using data from all runs
    fit_results = fit_discount_rate(data)
    k = fit_results['k']

    # Calculate subjective values
    data_with_sv = calculate_subjective_values(data, k)

    # Save the combined, trial-by-trial results
    output_dir = derivatives_dir / 'behavioral' / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{subject_id}_discounting_with_sv.tsv'
    data_with_sv.to_csv(output_path, sep='\t', index=False)
    
    print(f"Processed and saved combined data for {subject_id} from {len(event_files)} run(s)")
    
    # Return the summary results for aggregation
    return { 'subject_id': subject_id, **fit_results }

def main():
    """
    Main function to run the discount rate and subjective value calculations.
    """
    parser = argparse.ArgumentParser(description="Calculate discount rates and subjective values from BIDS events files.")
    parser.add_argument('--config', default='config/project_config.yaml', help='Path to the project configuration file.')
    parser.add_argument('--env', default='local', choices=['local', 'hpc'], help="Environment to use from the config file (e.g., 'local', 'hpc').")
    parser.add_argument('--subjects', nargs='+', help='A list of subject IDs to process. If not provided, all subjects will be processed.')
    parser.add_argument('--bids-dir', help='Override the BIDS data directory path from the config file.')
    parser.add_argument('--derivatives-dir', help='Override the derivatives directory path from the config file.')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    env_config = config[args.env]

    # Determine paths, allowing for overrides from command line
    bids_dir = Path(args.bids_dir) if args.bids_dir else Path(env_config['bids_dir'])
    derivatives_dir = Path(args.derivatives_dir) if args.derivatives_dir else Path(env_config['derivatives_dir'])
    onsets_dir = Path(env_config['onsets_dir'])

    # Add a check to ensure the BIDS and Onsets directories exist
    if not bids_dir.is_dir():
        raise FileNotFoundError(f"BIDS directory not found at the specified path: {bids_dir}")
    if not onsets_dir.is_dir():
        raise FileNotFoundError(f"Onsets directory not found at the specified path: {onsets_dir}")

    if args.subjects:
        subjects_to_process = args.subjects
    else:
        # Find all subjects in the BIDS directory if none are specified
        subjects_to_process = [d.name for d in bids_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]

    # --- Main processing loop ---
    all_fit_results = []
    for subject_id in subjects_to_process:
        subject_results = process_subject_data(subject_id, onsets_dir, derivatives_dir)
        if subject_results:
            all_fit_results.append(subject_results)

    # --- Save aggregated results ---
    if all_fit_results:
        summary_df = pd.DataFrame(all_fit_results)
        summary_output_dir = derivatives_dir / 'behavioral'
        summary_output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_output_dir / 'discounting_model_fits.tsv', sep='\t', index=False)
        print(f"\nSaved aggregated model fit results to {summary_output_dir}")


if __name__ == '__main__':
    main()
