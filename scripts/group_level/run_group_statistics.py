import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_1samp
import warnings

def load_config(config_file):
    """Load the project configuration from a YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def run_mvpa_group_stats(derivatives_dir, target):
    """
    Run group-level statistics for MVPA results.
    """
    print(f"--- Running Group-Level MVPA Statistics for Target: {target} ---")
    
    mvpa_dir = Path(derivatives_dir) / 'mvpa'
    behavioral_dir = Path(derivatives_dir) / 'behavioral'
    
    categorical_targets = ['choice']
    
    # --- 1. Calculate the appropriate chance level ---
    chance_level = 0.5
    if target in categorical_targets:
        subject_chance_levels = []
        # Find all subjects who have decoding results for this target
        subject_dirs = [p.parent for p in mvpa_dir.glob(f"sub-*/sub-*_target-{target}_decoding-scores.tsv")]
        
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            behavioral_file = behavioral_dir / subject_id / f"{subject_id}_discounting_with_sv.tsv"
            if behavioral_file.exists():
                df = pd.read_csv(behavioral_file, sep='\t')
                if target in df.columns:
                    # Calculate majority class proportion for this subject
                    value_counts = df[target].value_counts(normalize=True)
                    subject_chance_levels.append(value_counts.max())
        
        if subject_chance_levels:
            chance_level = np.mean(subject_chance_levels)
            print(f"Calculated group chance level (mean majority class) = {chance_level:.4f}")
        else:
            warnings.warn("Could not calculate subject-specific chance levels. Defaulting to 0.5.")
    else: # Regression analysis
        chance_level = 0.0
        print(f"Using chance level for regression (R^2 score) = {chance_level}")

    # --- 2. Aggregate individual subject scores ---
    decoding_files = list(mvpa_dir.glob(f"sub-*/sub-*_target-{target}_decoding-scores.tsv"))
    if not decoding_files:
        print(f"Error: No decoding result files found for target '{target}'.")
        return

    subject_mean_scores = []
    for f in decoding_files:
        scores = pd.read_csv(f, sep='\t')['scores']
        subject_mean_scores.append(scores.mean())
    
    subject_mean_scores = np.array(subject_mean_scores)

    # --- 3. Perform and report the t-test ---
    if len(subject_mean_scores) < 2:
        print("Error: Need at least two subjects to perform a group-level t-test.")
        return

    t_stat, p_value = ttest_1samp(subject_mean_scores, popmean=chance_level, alternative='greater')
    
    print("\n--- Group-Level MVPA Results ---")
    print(f"Number of subjects: {len(subject_mean_scores)}")
    print(f"Mean decoding score (accuracy/R^2): {subject_mean_scores.mean():.4f}")
    print(f"Standard deviation: {subject_mean_scores.std():.4f}")
    print(f"T-statistic vs. chance ({chance_level:.4f}): {t_stat:.4f}")
    print(f"P-value (one-tailed): {p_value:.4f}")
    print("---------------------------------\n")


def run_rsa_group_stats(derivatives_dir, metric):
    """
    Run group-level statistics for RSA results.
    """
    print(f"--- Running Group-Level RSA Statistics for Metric: {metric} ---")

    rsa_dir = Path(derivatives_dir) / 'rsa'
    chance_level = 0.0
    print(f"Using chance level for correlation = {chance_level}")

    # --- 1. Aggregate individual subject correlations ---
    rsa_files = list(rsa_dir.glob("sub-*/sub-*_rsa-results.tsv"))
    if not rsa_files:
        print("Error: No RSA result files found.")
        return

    subject_correlations = []
    for f in rsa_files:
        results_df = pd.read_csv(f, sep='\t')
        if metric in results_df['metric'].values:
            corr_value = results_df[results_df['metric'] == metric]['correlation'].iloc[0]
            subject_correlations.append(corr_value)
        else:
            warnings.warn(f"Metric '{metric}' not found in {f}. Skipping subject.")

    subject_correlations = np.array(subject_correlations)

    # --- 2. Perform and report the t-test ---
    if len(subject_correlations) < 2:
        print("Error: Need at least two subjects to perform a group-level t-test.")
        return
        
    t_stat, p_value = ttest_1samp(subject_correlations, popmean=chance_level, alternative='greater')

    print("\n--- Group-Level RSA Results ---")
    print(f"Number of subjects: {len(subject_correlations)}")
    print(f"Mean correlation (rho): {subject_correlations.mean():.4f}")
    print(f"Standard deviation: {subject_correlations.std():.4f}")
    print(f"T-statistic vs. chance ({chance_level:.4f}): {t_stat:.4f}")
    print(f"P-value (one-tailed): {p_value:.4f}")
    print("---------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Run group-level statistics on MVPA or RSA results.")
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to the project configuration file.')
    parser.add_argument('--env', type=str, default='local', choices=['local', 'hpc'], help='Environment to use from the config file.')
    parser.add_argument('--analysis-type', type=str, required=True, choices=['mvpa', 'rsa'], help='The type of analysis results to process.')
    parser.add_argument('--target', type=str, help='The target variable for MVPA analysis (e.g., choice, SVdiff). Required if --analysis-type is mvpa.')
    parser.add_argument('--metric', type=str, help="The RSA metric to test (e.g., 'whole_brain_rsa_SVdiff'). Required if --analysis-type is rsa.")
    
    args = parser.parse_args()

    if args.analysis_type == 'mvpa' and not args.target:
        parser.error("--target is required when --analysis-type is 'mvpa'")
    if args.analysis_type == 'rsa' and not args.metric:
        parser.error("--metric is required when --analysis-type is 'rsa'")

    config = load_config(args.config)
    paths = config[args.env]
    derivatives_dir = Path(paths['derivatives_dir'])

    if args.analysis_type == 'mvpa':
        run_mvpa_group_stats(derivatives_dir, args.target)
    elif args.analysis_type == 'rsa':
        run_rsa_group_stats(derivatives_dir, args.metric)

if __name__ == '__main__':
    main()
