import argparse
from pathlib import Path
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix
from nilearn.plotting import plot_stat_map, plot_glass_brain
import logging

from scripts.utils import load_config, setup_logging

def run_group_level_glm(config_path: str, env: str, contrast: str) -> None:
    """
    Runs a group-level (second-level) GLM analysis for a specified contrast.

    Args:
        config_path (str): Path to the project configuration file.
        env (str): The environment specified in the config file ('local' or 'hpc').
        contrast (str): The name of the contrast to analyze (e.g., 'SVchosen').
    """
    setup_logging()
    
    # --- 1. Load Configuration and Set Up Paths ---
    config = load_config(config_path)
    env_config = config[env]
    derivatives_dir = Path(env_config['derivatives_dir'])
    glm_output_dir = derivatives_dir / 'standard_glm'
    group_level_output_dir = derivatives_dir / 'group_level' / contrast
    group_level_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting group-level analysis for contrast: '{contrast}'")
    logging.info(f"First-level GLM outputs expected in: {glm_output_dir}")
    logging.info(f"Group-level outputs will be saved to: {group_level_output_dir}")

    # --- 2. Find All First-Level Contrast Maps ---
    # Find all subject-specific output directories from the first-level GLM
    subject_dirs = [d for d in glm_output_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    
    first_level_maps = []
    for sub_dir in subject_dirs:
        # The contrast map from the standard GLM is a z-map
        contrast_map_path = sub_dir / 'z_maps' / f'{contrast}_zmap.nii.gz'
        if contrast_map_path.exists():
            first_level_maps.append(str(contrast_map_path))
        else:
            logging.warning(f"Could not find contrast map for {sub_dir.name}. Skipping.")

    if len(first_level_maps) < 2:
        logging.error(f"Found fewer than 2 contrast maps ({len(first_level_maps)} found). Cannot run group-level analysis. Aborting.")
        return

    logging.info(f"Found {len(first_level_maps)} first-level contrast maps to include in the analysis.")

    # --- 3. Define and Fit the Second-Level Model ---
    # The design matrix for a simple group-level t-test is just an intercept
    design_matrix = pd.DataFrame([1] * len(first_level_maps), columns=['intercept'])

    # Initialize and fit the second-level GLM
    second_level_model = SecondLevelModel(smoothing_fwhm=8.0, verbose=1)
    second_level_model = second_level_model.fit(first_level_maps, design_matrix=design_matrix)

    # --- 4. Compute the Group-Level Contrast and Save Results ---
    # Compute the t-statistic for the intercept (i.e., the group mean)
    z_map = second_level_model.compute_contrast(output_type='z_score')
    
    output_path = group_level_output_dir / f'group_{contrast}_zmap.nii.gz'
    z_map.to_filename(output_path)
    logging.info(f"Saved group-level z-map to: {output_path}")

    # --- 5. Generate and Save Diagnostic Plots ---
    # Plot the unthresholded statistical map
    plot_stat_map(
        z_map,
        title=f"Group Level: {contrast} (Unthresholded)",
        output_file=group_level_output_dir / f'group_{contrast}_zmap_unthresholded.png',
        display_mode='z',
        cut_coords=8,
        colorbar=True
    )
    
    # Plot a glass brain view
    plot_glass_brain(
        z_map,
        title=f"Group Level: {contrast} (Glass Brain)",
        output_file=group_level_output_dir / f'group_{contrast}_glass_brain.png',
        colorbar=True,
        threshold=3.1 # Standard z-score threshold for p < 0.001
    )
    logging.info("Generated and saved diagnostic plots.")
    logging.info(f"Group-level analysis for contrast '{contrast}' complete.")

def main():
    parser = argparse.ArgumentParser(description="Run a second-level (group) GLM analysis.")
    parser.add_argument('--config', default='config/project_config.yaml', help='Path to the project configuration file.')
    parser.add_argument('--env', default='hpc', choices=['local', 'hpc'], help="Environment to use from the config file.")
    parser.add_argument('--contrast', required=True, help="The name of the contrast to analyze at the group level.")
    args = parser.parse_args()
    
    run_group_level_glm(args.config, args.env, args.contrast)

if __name__ == '__main__':
    main()
