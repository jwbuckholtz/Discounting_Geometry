import argparse
from pathlib import Path
import pandas as pd
import pingouin as pg
from scripts.utils import load_config, setup_logging
from typing import Dict, Any
import logging

def run_group_rsa_stats(derivatives_dir: Path) -> None:
    """
    Loads all subject RSA results and runs group-level stats.
    """
    logging.info("--- Running Group RSA Statistics ---")
    rsa_dir = derivatives_dir / 'rsa'
    
    # Correctly glob for all summary result files
    result_files = sorted(list(rsa_dir.glob("sub-*/summary_results/*_rsa-results.tsv")))
    
    if not result_files:
        raise FileNotFoundError(f"No RSA result files found in {rsa_dir}.")

    all_results_df = pd.concat([pd.read_csv(f, sep='\t') for f in result_files], ignore_index=True)

    # --- Perform t-test for each model and analysis type ---
    analyses = all_results_df['analysis'].unique()
    models = all_results_df['model'].unique()
    group_stats = []

    for analysis in analyses:
        for model in models:
            subset_df = all_results_df[(all_results_df['analysis'] == analysis) & (all_results_df['model'] == model)]
            if not subset_df.empty:
                # Average over folds for each subject
                subject_means = subset_df.groupby('subject_id')['correlation'].mean()
                
                ttest_res = pg.ttest(subject_means, 0, alternative='greater')
                ttest_res['analysis'] = analysis
                ttest_res['model'] = model
                ttest_res['mean_correlation'] = subject_means.mean()
                group_stats.append(ttest_res)

    summary_df = pd.concat(group_stats, ignore_index=True)
    logging.info("\n--- Group RSA Results ---")
    logging.info(summary_df[['analysis', 'model', 'mean_correlation', 'T', 'dof', 'p-val', 'cohen-d']])

    # --- Save Results ---
    output_dir = derivatives_dir / 'group_level'
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "group_rsa_stats.tsv", sep='\t', index=False)
    logging.info(f"\nSaved group RSA stats to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run group-level statistical analysis for RSA results.")
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to project config file')
    parser.add_argument('--env', type=str, required=True, choices=['local', 'hpc'], help='Environment')
    args = parser.parse_args()

    setup_logging()

    config = load_config(args.config)
    derivatives_dir = Path(config[args.env]['derivatives_dir'])

    run_group_rsa_stats(derivatives_dir)

if __name__ == "__main__":
    main()
