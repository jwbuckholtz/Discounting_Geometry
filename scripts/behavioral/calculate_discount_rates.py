import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

from scripts.utils import load_config, setup_logging

def hyperbolic_discount(delay: np.ndarray, amount: np.ndarray, k: float) -> np.ndarray:
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

def choice_probability(params: Tuple[float, float], S: np.ndarray, L: np.ndarray, D: np.ndarray) -> np.ndarray:
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
    # Clip the exponent to prevent overflow with very small tau values
    exponent = np.clip(-(sv_later - sv_sooner) / tau, -709, 709)
    prob_later = 1 / (1 + np.exp(exponent))
    return prob_later

def _validate_and_prepare_choice_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validates that the choice column exists and converts it to a 0/1 integer format.
    - Handles string labels ('larger_later', 'smaller_sooner') and numeric codes.
    - Raises ValueError if the column is missing or contains unexpected values.
    """
    if 'choice' not in data.columns:
        raise ValueError("The required column 'choice' was not found in the data.")

    # Convert to string to handle categorical/object dtypes robustly
    choice_col = data['choice'].astype(str).str.strip().str.lower()

    # Map string labels and numeric strings to 0/1
    choice_map = {'larger_later': 1, 'smaller_sooner': 0, '1': 1, '0': 0}
    data['choice'] = choice_col.map(choice_map)
    
    # Convert to numeric, coercing any non-mapped values to NaN
    data['choice'] = pd.to_numeric(data['choice'], errors='coerce')

    # Final validation: ensure the column contains only 0s and 1s (after dropping NaNs)
    if not data['choice'].dropna().isin([0, 1]).all():
        raise ValueError("Choice column contains values other than 0, 1, or recognized strings.")
        
    return data

def fit_discount_rate(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, float]:
    """
    Fits the hyperbolic discounting model to behavioral data to estimate k and tau.

    Args:
        data (pd.DataFrame): DataFrame with behavioral data for a single subject.
        params (dict): Dictionary of modeling parameters from the config file.

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

    # --- Data Preparation and Validation ---
    
    # 1. Validate that all required columns are present
    required_cols = ['small_amount', 'large_amount', 'later_delay', 'choice']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required behavioral columns: {', '.join(missing_cols)}")

    # 2. Validate and standardize the 'choice' column before dropping NaNs
    try:
        data = _validate_and_prepare_choice_data(data)
    except (ValueError, TypeError) as e:
        logging.error(f"Error processing 'choice' column: {e}")
        return {'k': np.nan, 'tau': np.nan, 'neg_log_likelihood': np.nan, 'pseudo_r2': np.nan}

    # 3. Drop rows with missing data in critical columns
    initial_rows = len(data)
    data.dropna(subset=required_cols, inplace=True)
    if len(data) < initial_rows:
        logging.info(f"Dropped {initial_rows - len(data)} rows with missing data.")
        
    # Abort if no valid trials remain after cleaning
    if data.empty:
        logging.warning("No valid trials remaining after removing rows with missing data. Aborting model fit.")
        return {'k': np.nan, 'tau': np.nan, 'neg_log_likelihood': np.nan, 'pseudo_r2': np.nan}
        
    # 4. Prepare numpy arrays for fitting
    S = data['small_amount'].values
    L = data['large_amount'].values
    D = data['later_delay'].values
    choices = data['choice'].values # Already 0/1 integer
    # --- End of Data Preparation ---

    # Parameters from config
    initial_params = params['initial_params']
    bounds = [tuple(params['k_bounds']), tuple(params['tau_bounds'])]

    # Minimize the negative log-likelihood using a more robust method
    result = minimize(neg_log_likelihood, initial_params, args=(S, L, D, choices),
                      method='L-BFGS-B',
                      bounds=bounds)

    if not result.success:
        logging.warning(f"Optimization failed. Reason: {result.message}")
        return {'k': np.nan, 'tau': np.nan, 'neg_log_likelihood': np.nan, 'pseudo_r2': np.nan}

    # Calculate pseudo-R-squared (McFadden's R-squared)
    ll_fit = -result.fun
    p_null = np.mean(choices)
    # Clip p_null to avoid log(0) errors for subjects with no variance in choice
    p_null = np.clip(p_null, 1e-10, 1 - 1e-10)
    ll_null = np.sum(choices * np.log(p_null) + (1 - choices) * np.log(1 - p_null))
    pseudo_r2 = 1 - (ll_fit / ll_null)
    
    fit_results = {
        'k': result.x[0],
        'tau': result.x[1],
        'neg_log_likelihood': result.fun,
        'pseudo_r2': pseudo_r2
    }
    return fit_results

def calculate_subjective_values(data: pd.DataFrame, k: float) -> pd.DataFrame:
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
    # This now works directly with the standardized 0/1 choice column
    is_later_chosen = data['choice'].astype(bool)
    
    data['SVchosen'] = np.where(is_later_chosen, sv_later, sv_sooner)
    data['SVunchosen'] = np.where(is_later_chosen, sv_sooner, sv_later)

    # Calculate sum and difference
    data['SVsum'] = sv_sooner + sv_later
    data['SVdiff'] = sv_later - sv_sooner # Larger-Later minus Smaller-Sooner

    return data

def process_subject_data(subject_id: str, onsets_dir: Path, derivatives_dir: Path, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes all behavioral data for a single subject, handling multiple runs.
    """
    
    # Corrected BIDS-compliant glob pattern to ensure exact subject matching.
    # Step 1: Find all event files starting with the exact subject ID.
    # The underscore is critical to prevent `sub-01` from matching `sub-010`.
    all_subject_files = onsets_dir.glob(f'{subject_id}_*_events.tsv')
    
    # Step 2: Filter for the specific task.
    event_files = sorted([f for f in all_subject_files if '_task-discountFix_' in f.name])
    
    if not event_files:
        logging.warning(f"No event files found for {subject_id}")
        return None
        
    # Load and concatenate data from all runs
    all_runs_data = []
    unassigned_run_counter = 1
    
    for event_file in event_files:
        # Extract run number from filename, with robust fallback
        run_part = event_file.stem.split('_run-')
        run = None
        if len(run_part) > 1:
            try:
                run = int(run_part[1].split('_')[0])
            except (ValueError, IndexError):
                logging.warning(f"Could not parse run number from '{event_file.name}'.")

        if run is None:
            # If no run number was parsed, assign a unique sequential number
            run = unassigned_run_counter
            unassigned_run_counter += 1
            logging.info(f"Assigning default run number {run} to '{event_file.name}'.")
        
        run_data = pd.read_csv(event_file, sep='\t')
        run_data['run'] = run
        all_runs_data.append(run_data)
    
    data = pd.concat(all_runs_data, ignore_index=True)

    # Fit the discount rate and temperature using data from all runs
    try:
        fit_results = fit_discount_rate(data, params)
    except ValueError as e:
        logging.error(f"Could not fit model for {subject_id}: {e}")
        # Create a result dict with NaNs to indicate failure for this subject
        fit_results = {
            'k': np.nan, 
            'tau': np.nan, 
            'neg_log_likelihood': np.nan, 
            'pseudo_r2': np.nan
        }
        
    k = fit_results['k']

    # Calculate subjective values
    data_with_sv = calculate_subjective_values(data, k)

    # Save the combined, trial-by-trial results
    output_dir = derivatives_dir / 'behavioral' / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{subject_id}_discounting_with_sv.tsv'
    data_with_sv.to_csv(output_path, sep='\t', index=False)
    
    logging.info(f"Processed and saved combined data for {subject_id} from {len(event_files)} run(s)")
    
    # Return the summary results for aggregation
    return { 'subject_id': subject_id, **fit_results }

def main() -> None:
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
    
    setup_logging()

    # Load configuration
    config = load_config(args.config)
    env_config = config[args.env]

    # Determine paths, allowing for overrides from command line
    bids_dir = Path(args.bids_dir) if args.bids_dir else Path(env_config['bids_dir'])
    derivatives_dir = Path(args.derivatives_dir) if args.derivatives_dir else Path(env_config['derivatives_dir'])
    onsets_dir = Path(env_config['onsets_dir'])
    analysis_params = config['analysis_params']['behavioral_modeling']

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
        subject_results = process_subject_data(subject_id, onsets_dir, derivatives_dir, analysis_params)
        if subject_results:
            all_fit_results.append(subject_results)

    # --- Save aggregated results ---
    if all_fit_results:
        summary_df = pd.DataFrame(all_fit_results)
        summary_output_dir = derivatives_dir / 'behavioral'
        summary_output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_output_dir / 'discounting_model_fits.tsv', sep='\t', index=False)
        logging.info(f"\nSaved aggregated model fit results to {summary_output_dir}")


if __name__ == '__main__':
    main()
