"""
Orchestrates the FINAL Monte Carlo simulation for all three comparison experiments.

This script runs the entire comparison over T multiple times with different
random seeds, collects all results, calculates statistics (mean, std),
and generates plots showing the mean performance with confidence bands.
"""

import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from experiments.run_final_comparison import run_final_comparison_experiment
from src.plotting import plot_mc_metric_comparison

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_monte_carlo_final_comparison(num_mc_runs: int = 10):
    """
    Performs the main Monte Carlo loop for all three methods.
    """
    print(f"--- Starting FINAL Monte Carlo Simulation with {num_mc_runs} runs ---")

    # A list to store the results DataFrame from each run
    all_dataframes = []

    # Use a ProcessPoolExecutor to run the Monte Carlo simulations in parallel.
    with ProcessPoolExecutor() as executor:
        seeds = range(num_mc_runs)
        
        results_iterator = executor.map(run_final_comparison_experiment, seeds)
        
        # Collect results as they are completed and add the run_id.
        for i, single_run_df in enumerate(tqdm(results_iterator, total=num_mc_runs, desc="Monte Carlo Runs")):
            if single_run_df is not None and not single_run_df.empty:
                single_run_df['run_id'] = i
                all_dataframes.append(single_run_df)

    # --- Process and analyze the collected results ---
    print("\nProcessing and analyzing all Monte Carlo results...")
    
    if not all_dataframes:
        print("Warning: No dataframes were collected. Aborting analysis.")
        return
        
    full_results_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save the raw data from all runs to a new file
    raw_results_path = os.path.join(RESULTS_DIR, "mc_final_comparison_raw.csv")
    full_results_df.to_csv(raw_results_path, index=False)
    print(f"-> Raw data from all {len(all_dataframes)} runs saved to {raw_results_path}")

    # Calculate statistics (mean and std) for each metric at each value of T
    summary_df = full_results_df.groupby('T').agg(['mean', 'std'])

    # Save the summary statistics to a new file
    summary_path = os.path.join(RESULTS_DIR, "mc_final_comparison_summary.csv")
    summary_df.to_csv(summary_path)
    print(f"-> Summary statistics saved to {summary_path}")

    # === Generate final plots with confidence bands for all three methods ===
    print("\nGenerating final comparison plots with confidence bands...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "mc_final_comparison")
    os.makedirs(figures_dir, exist_ok=True)
    
    # --- Plot configuration for Area ---
    area_configs = [
        {'col': 'dd_bounds_area', 'label': 'Data-Dependent', 'color': 'blue', 'marker': 'o'},
        {'col': 'bootstrap_area', 'label': 'Bootstrap', 'color': 'orange', 'marker': 'x'},
        {'col': 'set_membership_area', 'label': 'Set Membership (QMI)', 'color': 'green', 'marker': 's'}
    ]
    plot_mc_metric_comparison(summary_df, area_configs, 'T', 'Mean Area (log scale)', 
                              'Monte Carlo: Mean Area Comparison', 
                              os.path.join(figures_dir, "mc_final_comparison_area.png"))
    
    # --- Plot configuration for Worst-Case Deviation ---
    wcd_configs = [
        {'col': 'dd_bounds_wcd', 'label': 'Data-Dependent', 'color': 'blue', 'marker': 'o'},
        {'col': 'bootstrap_wcd', 'label': 'Bootstrap', 'color': 'orange', 'marker': 'x'},
        {'col': 'set_membership_wcd', 'label': 'Set Membership (QMI)', 'color': 'green', 'marker': 's'}
    ]
    plot_mc_metric_comparison(summary_df, wcd_configs, 'T', 'Mean WCD (log scale)', 
                              'Monte Carlo: Mean WCD Comparison', 
                              os.path.join(figures_dir, "mc_final_comparison_wcd.png"))

    # --- Plot configuration for Max Deviation in 'a' and 'b' ---
    max_dev_configs = [
        {'col': 'dd_bounds_max_dev_a', 'label': 'DD-Bounds (a)', 'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        {'col': 'bootstrap_max_dev_a', 'label': 'Bootstrap (a)', 'color': 'orange', 'marker': 'x', 'linestyle': '-'},
        {'col': 'set_membership_max_dev_a', 'label': 'Set Membership (a)', 'color': 'green', 'marker': 's', 'linestyle': '-'},
        {'col': 'dd_bounds_max_dev_b', 'label': 'DD-Bounds (b)', 'color': 'blue', 'marker': 'o', 'linestyle': '--'},
        {'col': 'bootstrap_max_dev_b', 'label': 'Bootstrap (b)', 'color': 'orange', 'marker': 'x', 'linestyle': '--'},
        {'col': 'set_membership_max_dev_b', 'label': 'Set Membership (b)', 'color': 'green', 'marker': 's', 'linestyle': '--'}
    ]
    plot_mc_metric_comparison(summary_df, max_dev_configs, 'T', 'Mean Max Deviation (log scale)', 
                              'Monte Carlo: Mean Max Deviation for "a" and "b"', 
                              os.path.join(figures_dir, "mc_final_comparison_dev_a_and_b.png"))

    print("\n--- FINAL Monte Carlo Simulation Finished Successfully! ---")

# Guard to allow direct execution for testing purposes
if __name__ == '__main__':
    run_monte_carlo_final_comparison()