import argparse
from pathlib import Path
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting
import matplotlib.pyplot as plt
from scripts.utils import load_config, setup_logging
from typing import Dict, Any
import logging

def run_group_glm(derivatives_dir: Path, contrast_id: str, n_subjects: int) -> None:
    """
    Runs a group-level (second-level) GLM analysis for a given contrast.
    """
    logging.info(f"--- Running Group-Level GLM for contrast: {contrast_id} ---")

    # --- 1. Find all first-level contrast maps ---
    first_level_dir = Path(derivatives_dir) / 'first_level_glms'
    contrast_maps = sorted(list(first_level_dir.glob(f"sub-*/sub-*_contrast-{contrast_id}_map.nii.gz")))
    
    if not contrast_maps:
        raise FileNotFoundError(f"No contrast maps found for '{contrast_id}' in {first_level_dir}")
    
    logging.info(f"Found {len(contrast_maps)} contrast maps for {len(n_subjects)} subjects.")
    if len(contrast_maps) != len(n_subjects):
        logging.warning("Number of contrast maps does not match number of subjects in BIDS dir.")


    # --- 2. Define and Fit Second-Level Model ---
    # A simple one-sample t-test against zero.
    design_matrix = pd.DataFrame([1] * len(contrast_maps), columns=['intercept'])
    
    second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
    second_level_model = second_level_model.fit(contrast_maps, design_matrix=design_matrix)

    # --- 3. Compute Group-Level Contrast ---
    z_map = second_level_model.compute_contrast(output_type='z_score')
    
    # --- 4. Save Results ---
    output_dir = Path(derivatives_dir) / 'group_level_glms'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    z_map_filename = output_dir / f"group_contrast-{contrast_id}_zmap.nii.gz"
    z_map.to_filename(z_map_filename)
    logging.info(f"Saved group z-map to {z_map_filename}")

    # --- 5. Create and Save Plots ---
    # Use nilearn's plotting functions to create a statistical map plot
    plotting.plot_stat_map(
        z_map,
        threshold=3.1,  # Corresponds to p < 0.001 uncorrected
        title=f"Group Level: {contrast_id}",
        output_file=output_dir / f"group_contrast-{contrast_id}_zmap.png"
    )
    # Create a glass brain plot
    plotting.plot_glass_brain(
        z_map,
        threshold=3.1,
        title=f"Group Level: {contrast_id} (Glass Brain)",
        output_file=output_dir / f"group_contrast-{contrast_id}_glassbrain.png"
    )
    plt.close('all') # Close plots to free memory
    logging.info(f"Saved result plots to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run group-level GLM analysis.")
    parser.add_argument('--config', type=str, default='config/project_config.yaml', help='Path to the project config file')
    parser.add_argument('--env', type=str, required=True, choices=['local', 'hpc'], help='Environment to run on')
    parser.add_argument('--contrast', type=str, required=True, help='The name of the contrast to analyze (e.g., SVdiff)')
    args = parser.parse_args()

    setup_logging()

    config = load_config(args.config)
    env_config = config[args.env]
    derivatives_dir = Path(env_config['derivatives_dir'])
    bids_dir = Path(env_config['bids_dir'])
    
    # Get subject list to verify against found contrast maps
    subjects = [s.name for s in bids_dir.glob('sub-*') if s.is_dir()]

    run_group_glm(derivatives_dir, args.contrast, subjects)
    logging.info("Group-level analysis finished successfully.")


if __name__ == "__main__":
    main()
