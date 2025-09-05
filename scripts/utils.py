import os
from pathlib import Path
import yaml
import pandas as pd

def load_config(config_path):
    """Loads the YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_lss_beta_maps(derivatives_dir, subject_id):
    """Finds the LSS beta maps file for a given subject."""
    betas_path = Path(derivatives_dir) / 'lss_betas' / subject_id / f'{subject_id}_lss_beta_maps.nii.gz'
    if not betas_path.exists():
        raise FileNotFoundError(f"LSS beta maps not found for {subject_id} at {betas_path}")
    return betas_path

def find_behavioral_sv_file(derivatives_dir, subject_id):
    """Finds the behavioral file with subjective values for a given subject."""
    events_path = Path(derivatives_dir) / 'behavioral' / subject_id / f'{subject_id}_discounting_with_sv.tsv'
    if not events_path.exists():
        raise FileNotFoundError(f"Behavioral SV file not found for {subject_id} at {events_path}")
    return events_path

def find_fmriprep_files(fmriprep_dir, subject_id):
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

def load_subject_data(config_path, env, subject_id):
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
