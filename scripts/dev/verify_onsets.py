import pandas as pd
from pathlib import Path
import sys
import yaml
import argparse

# Add project root to path to allow absolute imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from scripts.utils import find_behavioral_sv_file, find_subject_runs

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def inspect_onsets(subject_id, config_path, env):
    """
    Loads a subject's event file and prints the first 5 onsets for each run
    to verify if timing is session-wide or run-relative.
    """
    config = load_config(config_path)
    env_config = config[env]
    derivatives_dir = Path(env_config['derivatives_dir'])
    fmriprep_dir = Path(env_config['fmriprep_dir'])

    # Find a subject with multiple runs
    try:
        run_files = find_subject_runs(fmriprep_dir, subject_id)
        if len(run_files) < 2:
            print(f"Warning: Subject {subject_id} has fewer than 2 runs. Onset verification may be inconclusive.")
    except FileNotFoundError:
        print(f"Could not find BOLD data for subject {subject_id} in {fmriprep_dir}. Cannot determine number of runs.")
        return

    print(f"--- Inspecting onsets for {subject_id} ({len(run_files)} runs found) ---")
    
    try:
        events_file = find_behavioral_sv_file(derivatives_dir, subject_id)
        events_df = pd.read_csv(events_file, sep='\t')
    except FileNotFoundError:
        print(f"Could not find event file for {subject_id} at expected path containing: {derivatives_dir / 'behavioral' / subject_id}")
        return
        
    if 'run' not in events_df.columns:
        print("Error: 'run' column not found in event file. Cannot group by run.")
        return

    for run_num in sorted(events_df['run'].unique()):
        run_events = events_df[events_df['run'] == run_num]
        print(f"\n--- Run {run_num} ---")
        if not run_events.empty:
            print("First 5 onset times:")
            print(run_events['onset'].head().to_string(index=False))
        else:
            print("No events found for this run.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify event file onset timing for a specific subject.")
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., sub-s006)')
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to the project config file')
    parser.add_argument('--env', type=str, default='hpc', choices=['local', 'hpc'], help='Environment (local or hpc)')
    args = parser.parse_args()
    
    inspect_onsets(subject_id=args.subject, config_path=args.config, env=args.env)
