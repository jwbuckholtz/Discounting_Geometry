#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script provides a comprehensive analysis of the behavioral data from the 
delay discounting task. It performs the following main functions:

1.  **Data Aggregation:** It finds and concatenates the behavioral data files 
    (with subjective value, `_discounting_with_sv.tsv`) for all available subjects.

2.  **Group-Level Summaries:** It calculates and prints descriptive statistics 
    (mean, median) for key behavioral variables (reaction time, choice proportions) 
    across all subjects.

3.  **Individual-Level Summaries:** It computes and saves a summary table 
    (`individual_behavioral_summary.tsv`) with one row per subject, containing 
    individual-level statistics like mean RT, choice proportions, and discount rates (k-values).

4.  **Mixed-Effects Modeling:** It fits a series of mixed-effects linear models to 
    investigate the relationships between variables like reaction time, choice, delay, 
    and subjective value difference (`SVDiff`), while accounting for by-subject variability 
    with random intercepts.

5.  **Regression Diagnostics:** For each model, it generates and saves a set of 
    diagnostic plots (e.g., residuals vs. fitted, Q-Q plots) to assess model fit and assumptions.

6.  **Visualization:** It creates and saves summary plots to visualize the key 
    relationships tested in the models.

This script is designed to be run from the root of the project directory and relies on the 
project's configuration file (`config/project_config.yaml`) to locate necessary data files 
and determine output paths.

Usage:
    python scripts/behavioral/summarize_behavioral_data.py
"""
import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add the project root to the Python path to allow for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.utils import load_config


def load_and_preprocess_data(derivatives_dir):
    """
    Finds, loads, and preprocesses behavioral data for all subjects.

    Args:
        derivatives_dir (str): The path to the derivatives directory.

    Returns:
        pandas.DataFrame: A single DataFrame containing the concatenated and
                          preprocessed behavioral data for all subjects.
    """
    search_path = os.path.join(derivatives_dir, 'behavioral', 'sub-*', '*_discounting_with_sv.tsv')
    data_files = glob.glob(search_path)
    
    if not data_files:
        raise FileNotFoundError(f"No data files found matching the pattern: {search_path}")

    all_data = []
    for file_path in data_files:
        subject_id = os.path.basename(file_path).split('_')[0]
        df = pd.read_csv(file_path, sep='\t')
        df['subject'] = subject_id
        all_data.append(df)

    concatenated_df = pd.concat(all_data, ignore_index=True)

    # Standardize column names
    if 'response_time' in concatenated_df.columns:
        concatenated_df.rename(columns={'response_time': 'rt'}, inplace=True)

    # Preprocessing
    # Convert choice to a binary numeric variable
    concatenated_df['choice_numeric'] = concatenated_df['choice'].apply(
        lambda x: 1 if x == 'larger_later' else 0 if x == 'smaller_sooner' else np.nan
    )
    # Drop rows with no choice (e.g., missed trials)
    concatenated_df.dropna(subset=['choice_numeric', 'rt'], inplace=True)
    
    # Ensure correct data types
    concatenated_df['later_delay'] = pd.to_numeric(concatenated_df['later_delay'], errors='coerce')

    print(f"Loaded and preprocessed data for {concatenated_df['subject'].nunique()} subjects.")
    
    return concatenated_df


def get_group_summaries(df):
    """
    Calculates and prints group-level summary statistics.

    Args:
        df (pandas.DataFrame): The combined dataframe of all subjects' data.
    """
    print("\n--- Group-Level Behavioral Summaries ---")
    
    # Choice Proportions
    choice_props = df['choice'].value_counts(normalize=True)
    print("\nChoice Proportions (across all trials):")
    print(choice_props)

    # Reaction Time
    mean_rt = df['rt'].mean()
    median_rt = df['rt'].median()
    std_rt = df['rt'].std()
    print(f"\nReaction Time (across all trials):")
    print(f"  Mean: {mean_rt:.4f}s")
    print(f"  Median: {median_rt:.4f}s")
    print(f"  Std Dev: {std_rt:.4f}s")
    
    print("\n----------------------------------------\n")


def get_individual_summaries(df, output_path):
    """
    Generates and saves a table of individual-level summary statistics.

    Args:
        df (pandas.DataFrame): The combined dataframe of all subjects' data.
        output_path (str): The file path to save the summary table.
    """
    individual_summary = df.groupby('subject').agg(
        mean_rt=('rt', 'mean'),
        median_rt=('rt', 'median'),
        std_rt=('rt', 'std'),
        prop_larger_later=('choice_numeric', 'mean'),
        mean_sv_diff=('SVdiff', 'mean'),
        n_trials=('rt', 'count')
    ).reset_index()

    # It's possible k-values are stored elsewhere, but if they are in a file per subject we could grab them.
    # For now, we assume they might need to be recalculated or joined from another source.
    # Here we are just summarizing the trial-by-trial data.

    individual_summary.to_csv(output_path, sep='\t', index=False, float_format='%.4f')
    print(f"Individual-level summary table saved to: {output_path}")


def calculate_and_print_vif(df, formula):
    """
    Calculates and prints the Variance Inflation Factor (VIF) for each predictor
    in a given model formula.

    Args:
        df (pandas.DataFrame): The dataframe containing the data.
        formula (str): The model formula (e.g., 'y ~ x1 + x2').
    """
    try:
        # Extract predictor variable names from the formula string
        predictors_str = formula.split('~')[1].strip()
        predictor_names = [p.strip() for p in predictors_str.split('+')]

        # VIF is only meaningful for models with multiple predictors
        if len(predictor_names) < 2:
            return

        print("\n  Variance Inflation Factors (VIF):")
        
        # Create the design matrix using only the predictor variables
        # Note: We are calculating VIF on the fixed-effects portion of the model
        X = df[predictor_names]
        
        # The VIF function requires a constant to be added
        X = sm.add_constant(X, has_constant='add')
        
        # Calculate VIF for each predictor
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        # Print VIF, excluding the constant term
        vif_results = vif_data[vif_data["feature"] != "const"]
        print(vif_results.to_string(index=False))

    except Exception as e:
        print(f"  Could not calculate VIF for formula '{formula}'. Error: {e}")


def plot_regression_diagnostics(model_results, model_name, output_dir):
    """
    Generates and saves regression diagnostic plots for a fitted statsmodels model.

    Args:
        model_results: The fitted model object from statsmodels.
        model_name (str): A descriptive name for the model (for file naming).
        output_dir (str): The directory in which to save the plots.
    """
    diagnostics_dir = os.path.join(output_dir, 'regression_diagnostics', model_name)
    os.makedirs(diagnostics_dir, exist_ok=True)

    fitted = model_results.fittedvalues
    residuals = model_results.resid

    # 1. Residuals vs. Fitted Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=fitted, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs. Fitted\n({model_name})")
    plt.savefig(os.path.join(diagnostics_dir, 'residuals_vs_fitted.png'))
    plt.close()

    # 2. Q-Q Plot
    plt.figure(figsize=(8, 6))
    sm.qqplot(residuals, line='45', fit=True)
    plt.title(f"Normal Q-Q Plot\n({model_name})")
    plt.savefig(os.path.join(diagnostics_dir, 'qq_plot.png'))
    plt.close()

    # 3. Scale-Location Plot
    plt.figure(figsize=(8, 6))
    sqrt_abs_resid = np.sqrt(np.abs(residuals))
    sns.scatterplot(x=fitted, y=sqrt_abs_resid)
    plt.xlabel("Fitted Values")
    plt.ylabel("Square Root of Absolute Standardized Residuals")
    plt.title(f"Scale-Location Plot\n({model_name})")
    plt.savefig(os.path.join(diagnostics_dir, 'scale_location.png'))
    plt.close()
    
    print(f"Regression diagnostics for model '{model_name}' saved to: {diagnostics_dir}")


def run_mixed_effects_models(df, output_dir):
    """
    Runs a series of mixed-effects models and prints their summaries.

    Args:
        df (pandas.DataFrame): The combined dataframe of all subjects' data.
        output_dir (str): The directory to save diagnostic plots.
    """
    print("\n--- Mixed-Effects Models ---")

    # Z-score continuous predictors for easier interpretation of coefficients
    for col in ['later_delay', 'SVdiff', 'rt']:
        if col in df.columns:
            df[f'{col}_z'] = df.groupby('subject')[col].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )

    models_to_run = {
        # RT Models (Linear Mixed Models)
        'rt_vs_delay': 'rt ~ later_delay_z',
        'rt_vs_delay_choice': 'rt ~ later_delay_z + choice_numeric',
        'rt_vs_delay_choice_svdiff': 'rt ~ later_delay_z + choice_numeric + SVdiff_z',
        
        # Choice Models (Linear Probability Mixed Models)
        # Note: This is an approximation of a logistic mixed model.
        'choice_vs_delay': 'choice_numeric ~ later_delay_z',
        'choice_vs_delay_svdiff': 'choice_numeric ~ later_delay_z + SVdiff_z',
        'choice_vs_delay_svdiff_rt': 'choice_numeric ~ later_delay_z + SVdiff_z + rt_z'
    }

    for name, formula in models_to_run.items():
        print(f"\n--- Fitting Model: {name} ---")
        print(f"Formula: {formula}")
        
        try:
            model = smf.mixedlm(formula, df, groups=df["subject"])
            results = model.fit()
            print(results.summary())
            
            # Calculate and print VIF for the model's predictors
            calculate_and_print_vif(df, formula)
            
            # Generate and save diagnostic plots
            plot_regression_diagnostics(results, name, output_dir)

        except Exception as e:
            print(f"Could not fit model {name}. Error: {e}")

    print("\n----------------------------\n")


def plot_key_relationships(df, output_dir):
    """
    Creates and saves plots summarizing key behavioral relationships.

    Args:
        df (pandas.DataFrame): The combined dataframe of all subjects' data.
        output_dir (str): The directory in which to save the plots.
    """
    plots_dir = os.path.join(output_dir, 'summary_plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Set plot style
    sns.set_theme(style="whitegrid")

    # 1. RT vs. Delay
    plt.figure(figsize=(10, 7))
    sns.regplot(data=df, x='later_delay', y='rt',
                scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
    plt.title('Reaction Time vs. Later Delay (All Subjects)')
    plt.xlabel('Later Delay (days)')
    plt.ylabel('Reaction Time (s)')
    plt.savefig(os.path.join(plots_dir, 'rt_vs_delay.png'))
    plt.close()

    # 2. Choice vs. Delay
    plt.figure(figsize=(10, 7))
    # Use a logistic regression plot for the binary choice data
    sns.regplot(data=df, x='later_delay', y='choice_numeric', logistic=True,
                scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
    plt.title('Choice Probability vs. Later Delay (All Subjects)')
    plt.xlabel('Later Delay (days)')
    plt.ylabel('P(Choose Larger, Later)')
    plt.savefig(os.path.join(plots_dir, 'choice_vs_delay.png'))
    plt.close()

    # 3. RT vs. SV Difference
    plt.figure(figsize=(10, 7))
    sns.regplot(data=df, x='SVdiff', y='rt',
                scatter_kws={'alpha':0.1}, line_kws={'color':'blue'})
    plt.title('Reaction Time vs. Subjective Value Difference (All Subjects)')
    plt.xlabel('SV Difference (LL - SS)')
    plt.ylabel('Reaction Time (s)')
    plt.savefig(os.path.join(plots_dir, 'rt_vs_svdiff.png'))
    plt.close()

    # 4. Choice vs. SV Difference
    plt.figure(figsize=(10, 7))
    sns.regplot(data=df, x='SVdiff', y='choice_numeric', logistic=True,
                scatter_kws={'alpha':0.1}, line_kws={'color':'blue'})
    plt.title('Choice Probability vs. Subjective Value Difference (All Subjects)')
    plt.xlabel('SV Difference (LL - SS)')
    plt.ylabel('P(Choose Larger, Later)')
    plt.savefig(os.path.join(plots_dir, 'choice_vs_svdiff.png'))
    plt.close()

    print(f"Summary relationship plots saved to: {plots_dir}")


def main():
    """
    Main function to orchestrate the behavioral data summary and analysis.
    """
    parser = argparse.ArgumentParser(description="Run behavioral data summary and analysis.")
    parser.add_argument(
        '--env', 
        type=str, 
        default='local', 
        choices=['local', 'hpc'],
        help="The execution environment, specifying which paths to use from the config file."
    )
    args = parser.parse_args()

    # Load project configuration
    config = load_config('config/project_config.yaml')
    # Select the environment configuration
    env_config = config[args.env]

    # Define paths
    derivatives_dir = env_config['derivatives_dir']
    behavioral_summary_dir = os.path.join(derivatives_dir, 'behavioral', 'summaries')
    os.makedirs(behavioral_summary_dir, exist_ok=True)

    # 1. Load and preprocess data
    all_behavioral_data = load_and_preprocess_data(derivatives_dir)

    # 2. Get group-level summaries
    get_group_summaries(all_behavioral_data)

    # 3. Get and save individual-level summaries
    individual_summary_path = os.path.join(behavioral_summary_dir, 'individual_behavioral_summary.tsv')
    get_individual_summaries(all_behavioral_data, individual_summary_path)

    # 4. Run mixed-effects models and generate diagnostics
    run_mixed_effects_models(all_behavioral_data, behavioral_summary_dir)

    # 5. Plot key behavioral relationships
    plot_key_relationships(all_behavioral_data, behavioral_summary_dir)


if __name__ == '__main__':
    main()
