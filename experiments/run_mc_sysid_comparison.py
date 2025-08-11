"""
Orchestrates a Monte Carlo simulation for the SysId comparison experiment.

This script runs the entire comparison over T multiple times with different
random seeds, collects all results, calculates statistics (mean, std),
and generates plots showing the mean performance with confidence bands.
"""

import os
import pandas as pd
from tqdm import tqdm

# --- 1. Import the "worker" function and the plotter ---
from experiments.run_sysid_comparison import run_sysid_comparison_experiment
from src.plotting import plot_mc_metric_comparison

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_monte_carlo_sysid_comparison(num_mc_runs: int = 10):
    """
    Performs the main Monte Carlo loop.
    """
    print(f"--- Starting Monte Carlo Simulation with {num_mc_runs} runs ---")

    # A list to store the results DataFrame from each run
    all_dataframes = []

    for i in tqdm(range(num_mc_runs), desc="Monte Carlo Runs"):
        # Run the entire T-sweep experiment with a new master seed
        single_run_df = run_sysid_comparison_experiment(data_seed=i)
        if single_run_df is not None and not single_run_df.empty:
            single_run_df['run_id'] = i # Add an identifier for this run
            all_dataframes.append(single_run_df)
    
    # --- Process and analyze the collected results ---
    print("\nProcessing and analyzing all Monte Carlo results...")
    
    # Check if any results were collected before proceeding
    if not all_dataframes:
        print("Warning: No dataframes were collected. Aborting analysis.")
        return
        
    # Combine all individual DataFrames into one large DataFrame
    full_results_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save the raw data from all runs
    raw_results_path = os.path.join(RESULTS_DIR, "mc_sysid_comparison_raw.csv")
    full_results_df.to_csv(raw_results_path, index=False)
    print(f"-> Raw data from all {len(all_dataframes)} runs saved to {raw_results_path}")

    # Calculate statistics (mean and std) for each metric at each value of T
    summary_df = full_results_df.groupby('T').agg(['mean', 'std'])

    # Save the summary statistics
    summary_path = os.path.join(RESULTS_DIR, "mc_sysid_comparison_summary.csv")
    summary_df.to_csv(summary_path)
    print(f"-> Summary statistics saved to {summary_path}")


    print("\nGenerating final comparison plots with confidence bands...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "mc_sysid_comparison")
    os.makedirs(figures_dir, exist_ok=True)
    
    # --- Plot configuration for Area ---
    area_configs = [
        {'col': 'dd_bounds_area', 'label': 'Data-Dependent (Ellipse)', 'color': 'blue', 'marker': 'o'},
        {'col': 'bootstrap_area', 'label': 'Bootstrap (Rectangle)', 'color': 'orange', 'marker': 'x'}
    ]
    plot_mc_metric_comparison(summary_df, area_configs, 'T', 'Mean Area (log scale)', 
                              'Monte Carlo: Mean Area Comparison ', 
                              os.path.join(figures_dir, "mc_comparison_area.png"))
    
    # --- Plot configuration for Worst-Case Deviation ---
    wcd_configs = [
        {'col': 'dd_bounds_wcd', 'label': 'Data-Dependent (Ellipse)', 'color': 'blue', 'marker': 'o'},
        {'col': 'bootstrap_wcd', 'label': 'Bootstrap (Rectangle)', 'color': 'orange', 'marker': 'x'}
    ]
    plot_mc_metric_comparison(summary_df, wcd_configs, 'T', 'Mean Worst Case (log scale)', 
                              'Monte Carlo: Mean WCD Comparison', 
                              os.path.join(figures_dir, "mc_comparison_wcd.png"))

    # --- Plot configuration for Max Deviation in 'a' and 'b' ---
    max_dev_configs = [
        # Lines for parameter 'a' (solid)
        {'col': 'dd_bounds_max_dev_a', 'label': 'DD-Bounds (a)', 'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        {'col': 'bootstrap_max_dev_a', 'label': 'Bootstrap (a)', 'color': 'orange', 'marker': 'x', 'linestyle': '-'},
        # Lines for parameter 'b' (dashed)
        {'col': 'dd_bounds_max_dev_b', 'label': 'DD-Bounds (b)', 'color': 'blue', 'marker': 'o', 'linestyle': '--'},
        {'col': 'bootstrap_max_dev_b', 'label': 'Bootstrap (b)', 'color': 'orange', 'marker': 'x', 'linestyle': '--'},
    ]
    plot_mc_metric_comparison(summary_df, max_dev_configs, 'T', 'Mean Max Deviation (log scale)', 
                              'Monte Carlo: Mean Max Deviation for a and b', 
                              os.path.join(figures_dir, "mc_comparison_dev_a_and_b.png"))

    print("\n--- Monte Carlo Simulation Finished Successfully! ---")

# Guard to allow direct execution for testing purposes
if __name__ == '__main__':
    run_monte_carlo_sysid_comparison()