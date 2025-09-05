import argparse
from pathlib import Path
import pandas as pd
import pingouin as pg
from scripts.utils import load_config

def run_group_mvpa_stats(derivatives_dir, target):
    """
    Loads all subject MVPA results for a given target and runs group-level stats.
    """
    print(f"--- Running Group MVPA Statistics for target: {target} ---")
    mvpa_dir = derivatives_dir / 'mvpa'
    
    # Correctly glob for all ROI-specific result files for the target
    decoding_files = sorted(list(mvpa_dir.glob(f"sub-*/{target}/sub-*_target-{target}_roi-*_decoding-scores.tsv")))
    
    if not decoding_files:
        raise FileNotFoundError(f"No decoding result files found for target '{target}' in {mvpa_dir}.")
        
    # Load and concatenate all results
    all_results_df = pd.concat([pd.read_csv(f) for f in decoding_files], ignore_index=True)
    
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
    print("\n--- Group MVPA Results ---")
    print(summary_df[['roi', 'mean_score', 'T', 'dof', 'p-val', 'cohen-d']])
    
    # --- Save Results ---
    output_dir = derivatives_dir / 'group_level'
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / f"group_mvpa_stats_target-{target}.tsv", sep='\t', index=False)
    print(f"\nSaved group MVPA stats to {output_dir}")

def run_group_rsa_stats(derivatives_dir):
    """
    Loads all subject RSA results and runs group-level stats.
    """
    print("--- Running Group RSA Statistics ---")
    rsa_dir = derivatives_dir / 'rsa'
    
    # Correctly glob for all summary result files
    result_files = sorted(list(rsa_dir.glob("sub-*/summary_results/sub-*_rsa-results.tsv")))
    
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
    print("\n--- Group RSA Results ---")
    print(summary_df[['analysis', 'model', 'mean_correlation', 'T', 'dof', 'p-val', 'cohen-d']])

    # --- Save Results ---
    output_dir = derivatives_dir / 'group_level'
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "group_rsa_stats.tsv", sep='\t', index=False)
    print(f"\nSaved group RSA stats to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run group-level statistical analysis.")
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to project config file')
    parser.add_argument('--env', type=str, required=True, choices=['local', 'hpc'], help='Environment')
    parser.add_argument('--analysis-type', type=str, required=True, choices=['mvpa', 'rsa'], help='Type of analysis')
    parser.add_argument('--target', type=str, help='Target variable for MVPA stats (required if analysis-type is mvpa)')
    args = parser.parse_args()

    if args.analysis_type == 'mvpa' and not args.target:
        parser.error("--target is required when --analysis-type is 'mvpa'")

    config = load_config(args.config)
    derivatives_dir = Path(config[args.env]['derivatives_dir'])

    if args.analysis_type == 'mvpa':
        run_group_mvpa_stats(derivatives_dir, args.target)
    elif args.analysis_type == 'rsa':
        run_group_rsa_stats(derivatives_dir)

if __name__ == "__main__":
    main()
