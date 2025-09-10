import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from nilearn.interfaces.fmriprep import load_confounds_strategy
from typing import Dict, Any, List, Tuple
import nibabel as nib
import logging
from functools import reduce

def setup_logging():
    """Sets up a simple logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads the YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_lss_beta_maps(derivatives_dir: Path, subject_id: str) -> Path:
    """Finds the LSS beta maps file for a given subject."""
    betas_path = Path(derivatives_dir) / 'lss_betas' / subject_id / f'{subject_id}_lss_beta_maps.nii.gz'
    if not betas_path.exists():
        raise FileNotFoundError(f"LSS beta maps not found for {subject_id} at {betas_path}")
    return betas_path

def find_behavioral_sv_file(derivatives_dir: Path, subject_id: str) -> Path:
    """Finds the behavioral file with subjective values for a given subject."""
    events_path = Path(derivatives_dir) / 'behavioral' / subject_id / f'{subject_id}_discounting_with_sv.tsv'
    if not events_path.exists():
        raise FileNotFoundError(f"Behavioral SV file not found for {subject_id} at {events_path}")
    return events_path

def find_fmriprep_files(fmriprep_dir: Path, subject_id: str) -> Tuple[Path, Path, Path]:
    """
    Finds the necessary fMRIPrep output files for a given subject.
    This version is robust to session variability and uses a precise matching
    strategy based on the BOLD file's prefix.
    """
    fmriprep_path = Path(fmriprep_dir)
    subject_path = fmriprep_path / subject_id

    # Search in all session directories
    for ses_path in sorted(subject_path.glob('ses-*')):
        func_path = ses_path / 'func'
        if not func_path.exists():
            continue

        # 1. Find the BOLD file for our specific task 'discountFix'
        bold_files = list(func_path.glob(f'{subject_id}_{ses_path.name}_task-discountFix*_desc-preproc_bold.nii.gz'))
        if not bold_files:
            continue
        
        # In case of multiple runs, we'll use the first one found
        bold_path = bold_files[0]

        # 2. Extract the unique file prefix from the BOLD filename
        # This prefix is consistent across the BOLD and mask files
        file_prefix = bold_path.name.split('_desc-preproc_bold.nii.gz')[0]

        # 3. Use this precise prefix to find the corresponding mask
        mask_path = func_path / f'{file_prefix}_desc-brain_mask.nii.gz'

        # 4. Construct the confounds prefix separately, as it lacks the 'space' entity
        confounds_prefix = bold_path.name.split('_space-')[0]
        confounds_path = func_path / f'{confounds_prefix}_desc-confounds_timeseries.tsv'

        # 5. Check that all corresponding files actually exist before returning
        if mask_path.exists() and confounds_path.exists():
            print(f"Found matching file set in: {ses_path.name}")
            return bold_path, mask_path, confounds_path

    # If the loop completes without finding a complete set of files, raise an error
    raise FileNotFoundError(
        f"Could not find a complete set of BOLD, mask, and confounds files for "
        f"task 'discountFix' for {subject_id} in {fmriprep_dir}"
    )

def load_subject_data(config_path: Path, env: str, subject_id: str) -> Dict[str, Any]:
    """
    Loads all necessary files for a single subject for first-level modeling.
    
    Returns a dictionary containing paths and loaded dataframes.
    """
    config = load_config(config_path)
    env_config = config[env]
    derivatives_dir = Path(env_config['derivatives_dir'])
    fmriprep_dir = Path(env_config['fmriprep_dir'])

    # Find the required fMRIPrep and behavioral files
    bold_file, mask_file, confounds_file = find_fmriprep_files(fmriprep_dir, subject_id)
    events_file = find_behavioral_sv_file(derivatives_dir, subject_id)
    
    # Load behavioral data
    events_df = pd.read_csv(events_file, sep='\t')

    # Load confounds
    confounds_selected, _ = load_confounds_strategy(
        bold_file,
        denoise_strategy='simple',
        confounds_file=confounds_file
    )
    
    return {
        "subject_id": subject_id,
        "bold_file": bold_file,
        "mask_file": mask_file,
        "events_df": events_df,
        "confounds_selected": confounds_selected,
        "derivatives_dir": derivatives_dir
    }

def find_subject_runs(fmriprep_dir: Path, subject_id: str) -> List[Dict[str, str]]:
    """Finds all preprocessed BOLD files for all runs of the discountFix task."""
    fmriprep_path = Path(fmriprep_dir)
    subject_path = fmriprep_path / subject_id
    run_files = []

    for ses_path in sorted(subject_path.glob('ses-*')):
        func_path = ses_path / 'func'
        if not func_path.exists():
            continue
        
        # Find all BOLD files for the task
        bold_files = sorted(list(func_path.glob(f'{subject_id}_{ses_path.name}_task-discountFix*_desc-preproc_bold.nii.gz')))
        
        for bold_path in bold_files:
            file_prefix = bold_path.name.split('_desc-preproc_bold.nii.gz')[0]
            mask_path = func_path / f'{file_prefix}_desc-brain_mask.nii.gz'
            confounds_prefix = bold_path.name.split('_space-')[0]
            confounds_path = func_path / f'{confounds_prefix}_desc-confounds_timeseries.tsv'

            if mask_path.exists() and confounds_path.exists():
                run_files.append({
                    "bold": str(bold_path),
                    "mask": str(mask_path),
                    "confounds": str(confounds_path)
                })
    
    if not run_files:
        raise FileNotFoundError(f"No complete BOLD/mask/confounds sets found for task 'discountFix' for {subject_id}")
    
    return run_files

def load_concatenated_subject_data(config_path: Path, env: str, subject_id: str) -> Dict[str, Any]:
    """
    Loads and concatenates all functional data for a subject across all runs.
    """
    config = load_config(config_path)
    env_config = config[env]
    derivatives_dir = Path(env_config['derivatives_dir'])
    fmriprep_dir = Path(env_config['fmriprep_dir'])

    # 1. Find all runs
    run_files = find_subject_runs(fmriprep_dir, subject_id)
    print(f"Found {len(run_files)} runs for subject {subject_id}")

    # 2. Load and concatenate BOLD images and confounds
    bold_imgs = [run['bold'] for run in run_files]
    confounds_dfs = [pd.read_csv(run['confounds'], sep='\t') for run in run_files]
    run_numbers = [int(run['run_id']) for run in run_files] # Extract run numbers

    # 3. Load the run-aware behavioral file
    events_file = find_behavioral_sv_file(derivatives_dir, subject_id)
    events_df = pd.read_csv(events_file, sep='\t')
    
    # 4. Create the groups array for cross-validation
    groups = events_df['run'].values

    # 5. Use the mask from the first run (assuming all are aligned)
    mask_file = run_files[0]['mask']

    return {
        "subject_id": subject_id,
        "run_files": run_files, # Return the full run info
        "bold_imgs": bold_imgs,
        "run_numbers": run_numbers,
        "mask_file": mask_file,
        "events_df": events_df,
        "confounds_dfs": confounds_dfs,
        "groups": groups,
        "derivatives_dir": derivatives_dir
    }

def load_modeling_data(config_path: str, env: str, subject_id: str) -> Dict[str, Any]:
    """
    The definitive data loader for first-level modeling (GLM and LSS).

    - Finds all BOLD, confound, and mask files for a subject.
    - Loads the single, consolidated event file.
    - Prunes any runs that do not have corresponding events.
    - Standardizes confound columns across all valid runs using their UNION.
    - Returns a dictionary of all necessary, aligned data.
    """
    config = load_config(config_path)
    env_config = config[env]
    derivatives_dir = Path(env_config['derivatives_dir'])
    fmriprep_dir = Path(env_config['fmriprep_dir'])

    # 1. Find all potential runs and their files
    all_run_files = find_subject_runs(fmriprep_dir, subject_id)
    
    # 2. Load the single, consolidated event file
    events_file = find_behavioral_sv_file(derivatives_dir, subject_id)
    events_df = pd.read_csv(events_file, sep='\t')
    
    # 3. Prune runs that do not have corresponding events
    runs_with_events = events_df['run'].unique()
    valid_run_files = [run for run in all_run_files if int(run['run_id']) in runs_with_events]
    
    if len(valid_run_files) < len(all_run_files):
        logging.warning(f"Pruned {len(all_run_files) - len(valid_run_files)} run(s) with no matching events.")
        
    if not valid_run_files:
        raise FileNotFoundError(f"No valid runs with event data found for subject {subject_id}.")

    # 4. Load the data for the valid runs
    bold_imgs = [run['bold'] for run in valid_run_files]
    confounds_dfs = [pd.read_csv(run['confounds'], sep='\t') for run in valid_run_files]
    run_numbers = [int(run['run_id']) for run in valid_run_files]
    mask_file = valid_run_files[0]['mask']

    # 5. Standardize confounds using the UNION of columns
    if confounds_dfs:
        all_confound_cols = list(reduce(set.union, [set(df.columns) for df in confounds_dfs]))
        confounds_dfs = [df.reindex(columns=all_confound_cols, fill_value=0) for df in confounds_dfs]
        logging.info(f"Standardized confounds to {len(all_confound_cols)} columns across {len(valid_run_files)} runs.")

    return {
        "subject_id": subject_id,
        "bold_imgs": bold_imgs,
        "run_numbers": run_numbers,
        "mask_file": mask_file,
        "events_df": events_df,
        "confounds_dfs": confounds_dfs,
        "derivatives_dir": derivatives_dir
    }
