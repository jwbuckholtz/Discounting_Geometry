import argparse
import pandas as pd
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from scripts.utils import load_modeling_data, load_config, setup_logging
from typing import Dict, Any, List
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def _harmonize_confounds(confounds_dfs: List[pd.DataFrame], bold_imgs: List) -> List[pd.DataFrame] or None:
    """
    Harmonizes confound DataFrames to have identical column sets and orders.
    This prevents design matrix misalignment across runs.
    
    Args:
        confounds_dfs: List of confound DataFrames, one per run (may contain None entries)
        bold_imgs: List of BOLD images to get proper dimensions for missing confounds
        
    Returns:
        List of harmonized confound DataFrames with consistent columns, or None if no confounds
    """
    # Filter out None entries and empty DataFrames
    valid_confounds = [df for df in confounds_dfs if df is not None and not df.empty]
    
    if not valid_confounds:
        return None
        
    # Find the union of all columns across valid runs
    all_columns = set()
    for df in valid_confounds:
        all_columns.update(df.columns)
    
    # Sort columns for consistent ordering
    unified_columns = sorted(all_columns)
    
    # Harmonize each DataFrame, handling None entries with proper dimensions
    harmonized_dfs = []
    for i, df in enumerate(confounds_dfs):
        if df is None or df.empty:
            # Get the number of volumes from the corresponding BOLD image
            n_volumes = bold_imgs[i].shape[-1]
            
            # Create a DataFrame with correct dimensions (n_volumes x n_columns)
            harmonized_df = pd.DataFrame(
                data=np.zeros((n_volumes, len(unified_columns))),
                columns=unified_columns
            )
            logging.info(f"Run {i+1} has no confounds - creating zero-filled DataFrame with {n_volumes} volumes and {len(unified_columns)} columns")
        else:
            harmonized_df = df.copy()
            
            # Add missing columns with zeros
            for col in unified_columns:
                if col not in harmonized_df.columns:
                    harmonized_df[col] = 0.0
                    logging.info(f"Added missing confound column '{col}' to run {i+1} (filled with zeros)")
            
            # Reorder columns to match unified order
            harmonized_df = harmonized_df[unified_columns]
            
        harmonized_dfs.append(harmonized_df)
    
    logging.info(f"Harmonized {len(confounds_dfs)} confound DataFrames with {len(unified_columns)} columns: {unified_columns}")
    return harmonized_dfs

def _calculate_vif(X: np.ndarray, feature_names: list) -> Dict[str, float]:
    """
    Calculate Variance Inflation Factor for each regressor to detect multicollinearity.
    
    Args:
        X: Design matrix (n_samples x n_features)
        feature_names: Names of the features/regressors
        
    Returns:
        Dictionary mapping feature names to their VIF values
    """
    vif_dict = {}
    
    for i, feature in enumerate(feature_names):
        # Regress this feature against all others
        X_others = np.delete(X, i, axis=1)
        y = X[:, i]
        
        if X_others.shape[1] == 0:  # Only one feature
            vif_dict[feature] = 1.0
            continue
            
        try:
            # Fit regression
            reg = LinearRegression().fit(X_others, y)
            r2 = r2_score(y, reg.predict(X_others))
            
            # VIF = 1 / (1 - R²)
            vif = 1 / (1 - r2) if r2 < 0.999 else float('inf')
            vif_dict[feature] = vif
            
        except Exception:
            # If regression fails, assume no multicollinearity
            vif_dict[feature] = 1.0
    
    return vif_dict

def _validate_modulators_across_runs(events_df: pd.DataFrame, all_modulator_cols: list, run_numbers: list) -> list:
    """
    Validates modulators and returns those that are valid in at least one run.
    This is more flexible than requiring validity across ALL runs.
    
    Args:
        events_df: Complete events DataFrame with all runs
        all_modulator_cols: List of potential modulator columns
        run_numbers: List of run numbers to check
        
    Returns:
        List of modulator columns that are valid in at least one run
    """
    valid_modulators = []
    
    for modulator in all_modulator_cols:
        is_valid_in_any_run = False
        valid_runs = []
        
        for run_number in run_numbers:
            run_events = events_df[events_df['run'] == run_number]
            
            # Check if column exists and handle NaN values
            if modulator not in run_events.columns:
                continue  # Skip this run for this modulator
                
            run_events_copy = run_events.copy()
                
            # Handle all-NaN columns safely
            if run_events_copy[modulator].isna().all():
                continue  # Skip this run for this modulator
            
            # Count non-NaN entries - need at least 2 for variance calculation
            non_nan_count = run_events_copy[modulator].notna().sum()
            if non_nan_count <= 1:
                logging.debug(f"Modulator '{modulator}' has only {non_nan_count} non-NaN value(s) in run {run_number}")
                continue  # Skip this run for this modulator
                
            # Fill remaining NaNs with column mean and center
            col_mean = run_events_copy[modulator].mean()
            if pd.isna(col_mean):
                continue  # Skip this run for this modulator
                
            run_events_copy[modulator] = run_events_copy[modulator].fillna(col_mean)
            run_events_copy[modulator] -= col_mean
            
            # Check for zero variance after centering
            modulator_var = run_events_copy[modulator].var()
            if not (pd.isna(modulator_var) or np.isclose(modulator_var, 0)):
                is_valid_in_any_run = True
                valid_runs.append(run_number)
                
        if is_valid_in_any_run:
            valid_modulators.append(modulator)
            logging.info(f"Modulator '{modulator}' is valid in runs: {valid_runs}")
        else:
            logging.warning(f"Modulator '{modulator}' has no variance in any run. Will be excluded.")
            
    return valid_modulators

def prepare_run_events(run_events_df: pd.DataFrame, valid_modulator_cols: list) -> pd.DataFrame:
    """
    Prepares a single run's event DataFrame for the GLM in the format Nilearn expects
    for parametric modulation. Only processes modulators that have been validated across all runs.
    """
    # Clean and mean-center the validated modulator columns
    for col in valid_modulator_cols:
        if col not in run_events_df.columns:
            run_events_df[col] = 0
        else:
            # Handle NaN values safely (should be rare after validation)
            col_mean = run_events_df[col].mean()
            if not pd.isna(col_mean):
                run_events_df[col] = run_events_df[col].fillna(col_mean)
                run_events_df[col] -= col_mean
            else:
                # Fallback: set to zero if still all NaN
                run_events_df[col] = 0

    # Base events for the main effect of each trial
    events_base = run_events_df[['onset', 'duration']].copy()
    events_base['trial_type'] = 'mean'
    events_base['modulation'] = 1
    
    # Create new event rows for each validated parametric modulator
    events_modulated_list = [events_base]
    for modulator in valid_modulator_cols:
        modulator_events = run_events_df[['onset', 'duration']].copy()
        modulator_events['trial_type'] = modulator
        modulator_events['modulation'] = run_events_df[modulator]
        events_modulated_list.append(modulator_events)

    # Combine all event types into a single long-format DataFrame
    final_events_df = pd.concat(events_modulated_list, ignore_index=True)
    
    # Sort by onset to ensure chronological order (required by Nilearn)
    final_events_df = final_events_df.sort_values('onset').reset_index(drop=True)
    
    return final_events_df

def run_single_model_glm(subject_data: Dict[str, Any], params: Dict[str, Any], model_name: str, target_regressors: List[str]) -> None:
    """
    Runs a single GLM model with specified target regressors.
    
    Args:
        subject_data: Subject data dictionary
        params: Analysis parameters
        model_name: Name of the model for output organization
        target_regressors: List of regressors to include in this model
    """
    subject_id = subject_data['subject_id']
    bold_imgs = subject_data['bold_imgs']
    run_numbers = subject_data['run_numbers']
    events_df = subject_data['events_df']
    confounds_dfs = subject_data['confounds_dfs']
    derivatives_dir = subject_data['derivatives_dir']

    logging.info(f"--- Running {model_name} GLM for {subject_id} with regressors: {target_regressors} ---")
    
    # Create model-specific output directory
    output_dir = derivatives_dir / 'standard_glm' / subject_id / f'model-{model_name}'
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_params = params['analysis_params']
    glm = FirstLevelModel(
        t_r=analysis_params['t_r'],
        slice_time_ref=analysis_params['slice_time_ref'],
        hrf_model=analysis_params['glm']['hrf_model'],
        drift_model=analysis_params['glm']['drift_model'],
        smoothing_fwhm=analysis_params['smoothing_fwhm'],
        mask_img=subject_data['mask_file'],
        n_jobs=-1
    )
    
    # --- First Pass: Identify Runs with Events ---
    runs_with_events = []
    for run_number in run_numbers:
        run_events_df = events_df[events_df['run'] == run_number].copy()
        if not run_events_df.empty:
            runs_with_events.append(run_number)
        else:
            logging.warning(f"Run {run_number} has no events - will be skipped")
    
    if not runs_with_events:
        raise ValueError(f"No runs with events found for subject {subject_id}")
    
    # --- Validate Target Regressors Only on Runs with Events ---
    valid_modulator_cols = _validate_modulators_across_runs(events_df, target_regressors, runs_with_events)
    
    if len(valid_modulator_cols) < len(target_regressors):
        dropped = set(target_regressors) - set(valid_modulator_cols)
        logging.warning(f"Model {model_name}: Dropped invalid regressors: {dropped}")
    
    if not valid_modulator_cols:
        logging.warning(f"Model {model_name}: No valid regressors found - skipping")
        return

    # --- Prepare Events for Each Valid Run ---
    events_per_run = []
    valid_bold_imgs = []
    valid_confounds_dfs = []
    valid_run_numbers = []
    
    for i, run_number in enumerate(run_numbers):
        run_events_df = events_df[events_df['run'] == run_number].copy()
        
        # Skip runs with no events (already identified)
        if run_events_df.empty:
            continue
            
        # Check for completely missing essential columns that would prevent GLM
        required_cols = ['onset', 'duration']
        missing_required = [col for col in required_cols if col not in run_events_df.columns]
        if missing_required:
            logging.error(f"Run {run_number} missing required columns: {missing_required}, skipping")
            continue
            
        # Enhanced onset normalization with better heuristics and explicit timing support
        # Check if we have explicit run timing in the analysis parameters
        run_start_times = analysis_params.get('run_start_times', {})
        explicit_start = run_start_times.get(str(run_number))
        
        first_onset = run_events_df['onset'].min()
        last_onset = run_events_df['onset'].max()
        
        if explicit_start is not None:
            # Use explicit run start time from config (most reliable)
            logging.info(f"Using explicit start time {explicit_start}s for run {run_number}")
            run_events_df['onset'] -= explicit_start
        else:
            # Enhanced heuristic-based normalization
            onset_range = last_onset - first_onset
            
            # Multiple criteria for detecting session-relative timing
            needs_normalization = (
                first_onset > 300 or  # Original criterion
                (first_onset > 60 and onset_range < first_onset * 0.3) or  # Large start with compact range
                first_onset > 1800  # > 30 min definitely session-relative
            )
            
            if needs_normalization:
                logging.info(f"Normalizing session-relative onsets for run {run_number} "
                           f"(first: {first_onset:.1f}s, range: {onset_range:.1f}s)")
                run_events_df['onset'] -= first_onset
            else:
                # Preserve original timing but warn about potential issues
                logging.info(f"Preserving onsets for run {run_number} (first: {first_onset:.1f}s, last: {last_onset:.1f}s)")
                if first_onset > 10:  # Suspicious but not clearly session-relative
                    logging.warning(f"Run {run_number} onsets start at {first_onset:.1f}s - "
                                  "consider adding explicit 'run_start_times' to config if timing issues occur")
                
            # Always provide guidance on explicit timing
            if 'run_start_times' not in analysis_params:
                logging.info("For precise timing control, add 'run_start_times' to analysis_params in config")
        
        prepared_events = prepare_run_events(run_events_df, valid_modulator_cols)
        events_per_run.append(prepared_events)
        valid_bold_imgs.append(bold_imgs[i])
        valid_confounds_dfs.append(confounds_dfs[i])
        valid_run_numbers.append(run_number)

    if not events_per_run:
        raise ValueError(f"No valid runs with events found for subject {subject_id}")
        
    logging.info(f"Processing {len(events_per_run)} valid runs (skipped {len(run_numbers) - len(events_per_run)} empty runs)")
    
    # --- Harmonize Confounds ---
    harmonized_confounds = _harmonize_confounds(valid_confounds_dfs, valid_bold_imgs)
    
    if harmonized_confounds is None:
        logging.info("No confounds available - fitting GLM without confound regressors")
    else:
        logging.info(f"Using {len(harmonized_confounds)} harmonized confound DataFrames")
            
    # --- Fit the GLM ---
    glm.fit(valid_bold_imgs, events=events_per_run, confounds=harmonized_confounds)
    
    # --- Check for Multicollinearity with VIF Across All Runs ---
    if glm.design_matrices_:
        # Get only task regressors (exclude drift, constant, and confound regressors)
        design_cols = glm.design_matrices_[0].columns
        
        # More sophisticated filtering to exclude nuisance regressors
        nuisance_patterns = ['drift', 'constant', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 
                           'csf', 'white_matter', 'global_signal', 'framewise_displacement',
                           'a_comp_cor', 'cos', 'sin', 'dvars', 'std_dvars']
        
        task_regressors = []
        for col in design_cols:
            is_nuisance = any(pattern in col.lower() for pattern in nuisance_patterns)
            if not is_nuisance:
                task_regressors.append(col)
        
        if len(task_regressors) > 1:
            # Calculate VIF for each run and track maximum across runs
            max_vif_per_regressor = {}
            run_with_max_vif = {}
            
            for run_idx, design_matrix in enumerate(glm.design_matrices_):
                if all(reg in design_matrix.columns for reg in task_regressors):
                    # Calculate VIF for this run
                    task_design = design_matrix[task_regressors].values
                    vif_values = _calculate_vif(task_design, task_regressors)
                    
                    # Track maximum VIF per regressor across runs
                    for regressor, vif in vif_values.items():
                        if regressor not in max_vif_per_regressor or vif > max_vif_per_regressor[regressor]:
                            max_vif_per_regressor[regressor] = vif
                            run_with_max_vif[regressor] = run_idx + 1
            
            # Report VIF values and warn about high multicollinearity
            high_vif_regressors = []
            for regressor, max_vif in max_vif_per_regressor.items():
                if max_vif > 10:  # Common threshold for concerning multicollinearity
                    run_num = run_with_max_vif[regressor]
                    high_vif_regressors.append(f"{regressor}(VIF={max_vif:.2f} in run {run_num})")
                    
            if high_vif_regressors:
                logging.warning(f"Model {model_name}: High multicollinearity detected among task regressors: {', '.join(high_vif_regressors)}. "
                              "Consider orthogonalizing regressors or checking for redundant modulators.")
            else:
                overall_max_vif = max(max_vif_per_regressor.values()) if max_vif_per_regressor else 0
                logging.info(f"Model {model_name}: VIF check passed for {len(task_regressors)} task regressors across {len(glm.design_matrices_)} runs. Max VIF: {overall_max_vif:.2f}")
        else:
            logging.info(f"Model {model_name}: Only {len(task_regressors)} task regressor(s) found - VIF check not applicable")
    
    # --- Enforce Design Matrix Consistency (Including Column Order) ---
    if len(glm.design_matrices_) > 1:
        # Check that all design matrices have the same columns AND order
        first_columns_ordered = list(glm.design_matrices_[0].columns)
        first_columns_set = set(first_columns_ordered)
        inconsistent_runs = []
        
        for i, design_matrix in enumerate(glm.design_matrices_[1:], 1):
            current_columns_ordered = list(design_matrix.columns)
            current_columns_set = set(current_columns_ordered)
            
            # Check for missing/extra columns
            missing = first_columns_set - current_columns_set
            extra = current_columns_set - first_columns_set
            
            # Check for column order differences (even if same columns)
            order_mismatch = first_columns_ordered != current_columns_ordered
            
            if missing or extra or order_mismatch:
                inconsistent_runs.append({
                    'run': valid_run_numbers[i],
                    'missing': missing,
                    'extra': extra,
                    'order_mismatch': order_mismatch,
                    'expected_order': first_columns_ordered,
                    'actual_order': current_columns_ordered
                })
        
        # Enforce consistency - raise error if matrices differ
        if inconsistent_runs:
            error_msg = f"Design matrix inconsistencies detected for {subject_id} model {model_name}:\n"
            for issue in inconsistent_runs:
                run_num = issue['run']
                if issue['missing'] or issue['extra']:
                    error_msg += f"  Run {run_num}: missing={issue['missing']}, extra={issue['extra']}\n"
                if issue['order_mismatch']:
                    error_msg += f"  Run {run_num}: Column order mismatch\n"
                    error_msg += f"    Expected: {issue['expected_order']}\n"
                    error_msg += f"    Actual:   {issue['actual_order']}\n"
            error_msg += "This indicates problems with confound harmonization or event processing."
            raise ValueError(error_msg)
                    
        # Log the final design matrix structure
        logging.info(f"Model {model_name}: Design matrices consistent across {len(glm.design_matrices_)} runs with {len(first_columns_set)} columns in correct order: {first_columns_ordered}")
    else:
        # Single run case
        columns_ordered = list(glm.design_matrices_[0].columns)
        logging.info(f"Model {model_name}: Single run design matrix has {len(columns_ordered)} columns: {columns_ordered}")

    # --- Define and Compute Contrasts ---
    # Check for contrasts across ALL design matrices, not just the first one
    all_design_columns = set()
    for design_matrix in glm.design_matrices_:
        all_design_columns.update(design_matrix.columns)
    
    contrasts_to_compute = ['mean'] + valid_modulator_cols
    
    for contrast_name in contrasts_to_compute:
        if contrast_name in all_design_columns:
            logging.info(f"Model {model_name}: Computing contrast for '{contrast_name}'")
            
            # Compute contrast
            contrast_map = glm.compute_contrast(contrast_name, output_type='z_score')
            
            # Save contrast map
            contrast_path = output_dir / f'contrast-{contrast_name}_zmap.nii.gz'
            contrast_map.to_filename(str(contrast_path))
            logging.info(f"Model {model_name}: Saved contrast '{contrast_name}' to {contrast_path}")
        else:
            logging.warning(f"Model {model_name}: Contrast '{contrast_name}' not found in design matrix columns: {sorted(all_design_columns)}")
    
    logging.info(f"Model {model_name}: GLM analysis completed for {subject_id}")

def run_standard_glm_for_subject(subject_data: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Runs multiple separate GLM models to avoid multicollinearity issues.
    Each model focuses on a specific regressor of interest.
    """
    analysis_params = params['analysis_params']
    
    # Define the separate models to run
    model_specifications = analysis_params['glm'].get('model_specifications', {
        'choice': ['choice'],
        'value_chosen': ['SVchosen'],
        'value_unchosen': ['SVunchosen'], 
        'value_difference': ['SVdiff']
    })
    
    subject_id = subject_data['subject_id']
    logging.info(f"=== Running separate GLM models for {subject_id} ===")
    logging.info(f"Models to run: {list(model_specifications.keys())}")
    
    # Convert choice values to numeric if needed
    events_df = subject_data['events_df'].copy()
    if 'choice' in events_df.columns and events_df['choice'].dtype == 'object':
        choice_mapping = {'smaller_sooner': 0, 'larger_later': 1}
        events_df['choice'] = events_df['choice'].map(choice_mapping)
        logging.info("Converted choice column from text to numeric (0=smaller_sooner, 1=larger_later)")
        
        # Update subject_data with the converted events
        subject_data = subject_data.copy()
        subject_data['events_df'] = events_df
    
    # Run each model separately
    successful_models = []
    failed_models = []
    
    for model_name, target_regressors in model_specifications.items():
        try:
            run_single_model_glm(subject_data, params, model_name, target_regressors)
            successful_models.append(model_name)
        except Exception as e:
            logging.error(f"Model {model_name} failed: {e}")
            failed_models.append((model_name, str(e)))
    
    # Summary
    logging.info(f"=== GLM Analysis Summary for {subject_id} ===")
    logging.info(f"Successful models ({len(successful_models)}): {successful_models}")
    if failed_models:
        logging.warning(f"Failed models ({len(failed_models)}): {[name for name, _ in failed_models]}")
        for name, error in failed_models:
            logging.warning(f"  {name}: {error}")
    
    if not successful_models:
        raise ValueError(f"All GLM models failed for subject {subject_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standard GLM analysis for a subject")
    parser.add_argument('config_path', help='Path to configuration file')
    parser.add_argument('env', help='Environment (e.g., local, hpc)')
    parser.add_argument('subject_id', help='Subject ID to process')
    
    args = parser.parse_args()
    
    # Load configuration and setup logging
    config = load_config(args.config_path, args.env)
    setup_logging(config)
    
    # Load subject data
    subject_data = load_modeling_data(args.config_path, args.env, args.subject_id)
    
    # Run GLM analysis
    run_standard_glm_for_subject(subject_data, config)
