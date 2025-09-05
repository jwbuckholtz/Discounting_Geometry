import argparse
from pathlib import Path
import pandas as pd
import pingouin as pg
from scripts.utils import load_config, setup_logging
from typing import Dict, Any
import logging

def run_group_mvpa_stats(derivatives_dir: Path, target: str) -> None:
    """
    Loads all subject MVPA results for a given target and runs group-level stats.
    """
    logging.info(f"--- Running Group MVPA Statistics for target: {target} ---")
    mvpa_dir = derivatives_dir / 'mvpa'
    
    # Correctly glob for all ROI-specific result files for the target
    decoding_files = sorted(list(mvpa_dir.glob(f"sub-*/**/sub-*_target-{target}_roi-*_decoding-scores.tsv")))
    
    if not decoding_files:
        raise FileNotFoundError(f"No decoding result files found for target '{target}' in {mvpa_dir}.")
        
    # Load and concatenate all results
    all_results_df = pd.concat([pd.read_csv(f, sep='\t') for f in decoding_files], ignore_index=True)
    
    # Calculate mean accuracy for each subject and ROI
    subject_means = all_results_df.groupby(['subject_id', 'roi'])['scores'].mean().reset_index()
    
    # --- Perform t-test for each ROI ---
    rois = subject_means['roi'].unique()
    group_stats = []

    for roi in rois:
        roi_scores = subject_means[subject_means['roi'] == roi]['scores']
        # Determine chance level based on target
        chance_level = 0.5 if target == 'choice' else 0.0
        
        ttest_res = pg.ttest(roi_scores, chance_level, alternative='greater')
        ttest_res['roi'] = roi
        ttest_res['mean_score'] = roi_scores.mean()
        group_stats.append(ttest_res)
        
    summary_df = pd.concat(group_stats, ignore_index=True)
    logging.info("\n--- Group MVPA Results ---")
    logging.info(summary_df[['roi', 'mean_score', 'T', 'dof', 'p-val', 'cohen-d']])
    
    # --- Save Results ---
    output_dir = derivatives_dir / 'group_level'
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / f"group_mvpa_stats_target-{target}.tsv", sep='\t', index=False)
    logging.info(f"\nSaved group MVPA stats to {output_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run group-level statistical analysis for MVPA results.")
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to project config file')
    parser.add_argument('--env', type=str, required=True, choices=['local', 'hpc'], help='Environment')
    parser.add_argument('--target', type=str, required=True, help='Target variable for MVPA stats')
    args = parser.parse_args()

    setup_logging()

    config = load_config(args.config)
    derivatives_dir = Path(config[args.env]['derivatives_dir'])

    run_group_mvpa_stats(derivatives_dir, args.target)

if __name__ == "__main__":
    main()
