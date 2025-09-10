import argparse
from pathlib import Path
import pandas as pd
import pingouin as pg
from scripts.utils import load_config, setup_logging
from typing import Dict, Any
import logging

def run_group_mvpa_stats(derivatives_dir: Path) -> None:
    """
    Loads all subject MVPA results and runs group-level stats,
    intelligently selecting the null hypothesis based on the scoring metric.
    """
    logging.info(f"--- Running Group MVPA Statistics ---")
    mvpa_dir = derivatives_dir / 'mvpa'
    
    # Glob for all possible result files from all subjects
    decoding_files = sorted(list(mvpa_dir.glob(f"sub-*/**/*decoding-scores.tsv")))
    
    if not decoding_files:
        raise FileNotFoundError(f"No decoding result files found in {mvpa_dir}.")
        
    all_results_df = pd.concat([pd.read_csv(f, sep='\t') for f in decoding_files], ignore_index=True)
    
    # Calculate mean score for each subject, roi, target, and scoring metric
    subject_means = all_results_df.groupby(['subject_id', 'roi', 'target_variable', 'scoring']).scores.mean().reset_index()
    
    # --- Perform t-test for each ROI, Target, and Scoring combination ---
    group_stats = []
    for (roi, target, scoring), data in subject_means.groupby(['roi', 'target_variable', 'scoring']):
        scores = data['scores']
        
        # Determine the correct chance level for the t-test based on the metric
        popmean = 0.0 # Default for metrics like R^2 where chance is 0
        if scoring == 'accuracy':
            # This could be elaborated with n_classes if available, but 0.5 is a robust default for binary
            popmean = 0.5
        elif scoring not in ['r2', 'roc_auc']:
            logging.warning(f"T-test for scoring metric '{scoring}' is compared against 0. This may not be appropriate.")

        ttest_res = pg.ttest(scores, popmean, alternative='greater')
        ttest_res['roi'] = roi
        ttest_res['target_variable'] = target
        ttest_res['scoring'] = scoring
        ttest_res['mean_score'] = scores.mean()
        group_stats.append(ttest_res)
        
    if not group_stats:
        logging.warning("No data available to perform group-level statistics.")
        return

    summary_df = pd.concat(group_stats, ignore_index=True)
    logging.info("\n--- Group MVPA Results ---")
    print(summary_df[['roi', 'target_variable', 'scoring', 'mean_score', 'T', 'dof', 'p-val', 'cohen-d']].round(3).to_string(index=False))
    
    # --- Save Results ---
    output_dir = derivatives_dir / 'group_level'
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "group_mvpa_stats_summary.tsv", sep='\t', index=False)
    logging.info(f"\nSaved group MVPA stats to {output_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run group-level statistical analysis for all MVPA results.")
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to project config file')
    parser.add_argument('--env', type=str, required=True, choices=['local', 'hpc'], help='Environment')
    args = parser.parse_args()

    setup_logging()

    config = load_config(args.config)
    derivatives_dir = Path(config[args.env]['derivatives_dir'])

    run_group_mvpa_stats(derivatives_dir)

if __name__ == "__main__":
    main()
