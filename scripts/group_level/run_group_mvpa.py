import argparse
from pathlib import Path
import pandas as pd
import pingouin as pg
from scripts.utils import load_config, setup_logging
from typing import Dict, Any
import logging

def run_group_mvpa_stats(derivatives_dir: Path, analysis_params: Dict[str, Any]) -> None:
    """
    Loads all subject MVPA results and runs group-level stats,
    intelligently selecting the null hypothesis based on the scoring metric.
    """
    logging.info(f"--- Running Group MVPA Statistics ---")
    mvpa_dir = derivatives_dir / 'mvpa'
    
    # Glob for all possible result files
    decoding_files = sorted(list(mvpa_dir.glob(f"sub-*/**/*decoding-scores.tsv")))
    
    if not decoding_files:
        raise FileNotFoundError(f"No decoding result files found in {mvpa_dir}.")
        
    all_results_df = pd.concat([pd.read_csv(f, sep='\t') for f in decoding_files], ignore_index=True)
    
    # We need to find out which targets are classification vs regression
    class_targets = analysis_params['mvpa']['classification']['target_variables']
    
    # Calculate mean score for each subject, roi, and target
    subject_means = all_results_df.groupby(['subject_id', 'roi', 'target_variable', 'scoring']).scores.mean().reset_index()
    
    # --- Perform t-test for each ROI and Target combination ---
    group_stats = []
    for (roi, target, scoring), data in subject_means.groupby(['roi', 'target_variable', 'scoring']):
        scores = data['scores']
        
        # Determine the correct chance level for the t-test
        popmean = 0.0 # Default for metrics like R^2
        if scoring == 'accuracy':
            # Assuming binary classification for now
            popmean = 0.5
        elif scoring not in ['r2', 'roc_auc']:
            logging.warning(f"T-test for scoring metric '{scoring}' is compared against 0, but this may not be appropriate. "
                            "Consider a permutation test for a more accurate null hypothesis.")

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

    run_group_mvpa_stats(derivatives_dir, config['analysis_params'])

if __name__ == "__main__":
    main()
