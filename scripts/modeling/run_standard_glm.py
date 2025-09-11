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
                
            # Fill remaining NaNs with column mean and center
            col_mean = run_events_copy[modulator].mean()
            if pd.isna(col_mean):
                continue  # Skip this run for this modulator
                
            run_events_copy[modulator] = run_events_copy[modulator].fillna(col_mean)
            run_events_copy[modulator] -= col_mean
            
            # Check for zero variance after centering
            if not np.isclose(run_events_copy[modulator].var(), 0):
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

def run_standard_glm_for_subject(subject_data: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Runs a standard GLM with parametric modulators across one or more runs.
    """
    subject_id = subject_data['subject_id']
    bold_imgs = subject_data['bold_imgs']
    run_numbers = subject_data['run_numbers']
    events_df = subject_data['events_df']
    confounds_dfs = subject_data['confounds_dfs']
    derivatives_dir = subject_data['derivatives_dir']

    logging.info(f"--- Running Standard GLM for {subject_id} on {len(bold_imgs)} run(s) ---")
    output_dir = derivatives_dir / 'standard_glm' / subject_id
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
    
    # --- Validate Modulators Only on Runs with Events ---
    all_modulator_cols = analysis_params['glm']['parametric_modulators']
    valid_modulator_cols = _validate_modulators_across_runs(events_df, all_modulator_cols, runs_with_events)
    
    if len(valid_modulator_cols) < len(all_modulator_cols):
        dropped = set(all_modulator_cols) - set(valid_modulator_cols)
        logging.info(f"Dropped modulators that were invalid across runs with events: {dropped}")
    
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
            
        # Explicit onset normalization based on run timing expectations
        first_onset = run_events_df['onset'].min()
        last_onset = run_events_df['onset'].max()
        
        # Check if we have explicit run start time information in the config
        # For now, use a more conservative approach that preserves intentional timing
        
        # Enhanced onset normalization with better heuristics and explicit timing support
        # Check if we have explicit run timing in the analysis parameters
        run_start_times = analysis_params.get('run_start_times', {})
        explicit_start = run_start_times.get(str(run_number))
        
        if explicit_start is not None:
            # Use explicit run start time from config (most reliable)
            logging.info(f"Using explicit start time {explicit_start}s for run {run_number}")
            run_events_df['onset'] -= explicit_start
        else:
            # Enhanced heuristic-based normalization
            onset_range = last_onset - first_onset
            
            # Multiple criteria for detecting session-relative timing:
            # 1. First onset > 300s (original criterion)
            # 2. Large onset range suggesting session timing
            # 3. First onset significantly larger than typical run duration
            
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
    
    # Check if we have any valid runs left
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
    
    # --- Check for Multicollinearity with VIF ---
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
            # Calculate VIF for task regressors only
            task_design = glm.design_matrices_[0][task_regressors].values
            vif_values = _calculate_vif(task_design, task_regressors)
            
            # Report VIF values and warn about high multicollinearity
            high_vif_regressors = []
            for regressor, vif in vif_values.items():
                if vif > 10:  # Common threshold for concerning multicollinearity
                    high_vif_regressors.append(f"{regressor}(VIF={vif:.2f})")
                    
            if high_vif_regressors:
                logging.warning(f"High multicollinearity detected among task regressors: {', '.join(high_vif_regressors)}. "
                              "Consider orthogonalizing regressors or checking for redundant modulators.")
            else:
                logging.info(f"VIF check passed for {len(task_regressors)} task regressors. Max VIF: {max(vif_values.values()):.2f}")
        else:
            logging.info(f"Only {len(task_regressors)} task regressor(s) found - VIF check not applicable")
    
    # --- Enforce Design Matrix Consistency ---
    if len(glm.design_matrices_) > 1:
        # Check that all design matrices have the same columns
        first_columns = set(glm.design_matrices_[0].columns)
        inconsistent_runs = []
        
        for i, design_matrix in enumerate(glm.design_matrices_[1:], 1):
            current_columns = set(design_matrix.columns)
            if current_columns != first_columns:
                missing = first_columns - current_columns
                extra = current_columns - first_columns
                inconsistent_runs.append({
                    'run': valid_run_numbers[i],
                    'missing': missing,
                    'extra': extra
                })
        
        # Enforce consistency - raise error if matrices differ
        if inconsistent_runs:
            error_msg = f"Design matrix inconsistencies detected for {subject_id}:\n"
            for issue in inconsistent_runs:
                error_msg += f"  Run {issue['run']}: missing={issue['missing']}, extra={issue['extra']}\n"
            error_msg += "This indicates problems with confound harmonization or event processing."
            raise ValueError(error_msg)
                    
        # Log the final design matrix structure
        logging.info(f"Design matrices consistent across {len(glm.design_matrices_)} runs with {len(first_columns)} columns: {sorted(first_columns)}")
    else:
        # Single run case
        columns = set(glm.design_matrices_[0].columns)
        logging.info(f"Single run design matrix has {len(columns)} columns: {sorted(columns)}")

    # --- Define and Compute Contrasts ---
    # Check for contrasts across ALL design matrices, not just the first one
    all_design_columns = set()
    for design_matrix in glm.design_matrices_:
        all_design_columns.update(design_matrix.columns)
    
    contrasts_to_compute = ['mean'] + valid_modulator_cols
    
    for contrast_name in contrasts_to_compute:
        if contrast_name not in all_design_columns:
            logging.warning(
                f"Contrast '{contrast_name}' not found in any design matrix. Skipping."
            )
            continue
            
        # Check if the contrast exists in ALL runs (required for valid contrast)
        missing_in_runs = []
        for i, design_matrix in enumerate(glm.design_matrices_):
            if contrast_name not in design_matrix.columns:
                missing_in_runs.append(i + 1)
                
        if missing_in_runs:
            logging.warning(
                f"Contrast '{contrast_name}' missing in run(s) {missing_in_runs}. Skipping."
            )
            continue
            
        # The regressor names are now directly the trial_type names
        z_map = glm.compute_contrast(contrast_name, output_type='z_score')
        
        contrast_filename = output_dir / f"contrast-{contrast_name}_zmap.nii.gz"
        z_map.to_filename(contrast_filename)
        logging.info(f"Saved z-map to: {contrast_filename.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standard GLM for a single subject.")
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to the project config file')
    parser.add_argument('--env', type=str, required=True, choices=['local', 'hpc'], help='Environment to run on')
    parser.add_argument('--subject', type=str, required=True, help='The subject ID to process')
    args = parser.parse_args()

    setup_logging()

    # Load the full config
    config = load_config(args.config)
    
    # Load all data for the subject using the new, definitive utility function
    subject_data = load_modeling_data(args.config, args.env, args.subject)
    
    # Pass the full config to the analysis function
    run_standard_glm_for_subject(subject_data, config)


if __name__ == "__main__":
    main()
