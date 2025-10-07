"""
Orchestrates a Monte Carlo simulation for the extended SysId comparison experiment.
"""
import os
import pandas as pd
from tqdm import tqdm

from old.run_sysid_comparison import run_sysid_methods_comparison
from src.plotting import plot_mc_metric_comparison

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

 
def run_monte_carlo_sysid_methods(num_mc_runs: int = 10):
    """
    Performs the main Monte Carlo loop for the three SysId methods.
    """
    print(f"--- Starting Monte Carlo for 3 SysId Methods with {num_mc_runs} runs ---")

    all_dataframes = []
    for i in tqdm(range(num_mc_runs), desc="Monte Carlo Runs"):
        single_run_df = run_sysid_methods_comparison(data_seed=i)
        if single_run_df is not None and not single_run_df.empty:
            single_run_df['run_id'] = i
            all_dataframes.append(single_run_df)
    
    if not all_dataframes:
        print("Warning: No dataframes were collected. Aborting analysis.")
        return
        
    full_results_df = pd.concat(all_dataframes, ignore_index=True)
    
    raw_results_path = os.path.join(RESULTS_DIR, "mc_sysid_methods_raw.csv")
    full_results_df.to_csv(raw_results_path, index=False)
    print(f"\n-> Raw data saved to {raw_results_path}")

    summary_df = full_results_df.groupby('T').agg(['mean', 'std'])
    summary_path = os.path.join(RESULTS_DIR, "mc_sysid_methods_summary.csv")
    summary_df.to_csv(summary_path)
    print(f"-> Summary statistics saved to {summary_path}")

    # --- Generate final plots with confidence bands ---
    print("\nGenerating final comparison plots...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "mc_sysid_methods")
    os.makedirs(figures_dir, exist_ok=True)
    
    # --- Plot configuration for Area ---
    area_configs = [
        {'col': 'dd_bounds_area', 'label': 'Data-Dependent (I.I.D.)', 'color': 'blue', 'marker': 'o'},
        {'col': 'bootstrap_iid_area', 'label': 'Bootstrap (I.I.D.)', 'color': 'red', 'marker': 'v'},
        {'col': 'bootstrap_ts_area', 'label': 'Bootstrap (Time-Series)', 'color': 'orange', 'marker': 'x'}
    ]
    plot_mc_metric_comparison(summary_df, area_configs, 'T', 'Mean Area (log scale)', 'MC: Mean Area Comparison', os.path.join(figures_dir, "mc_comparison_area.png"))
    
    # --- Plot configuration for WCD ---
    wcd_configs = [
        {'col': 'dd_bounds_wcd', 'label': 'Data-Dependent (I.I.D.)', 'color': 'blue', 'marker': 'o'},
        {'col': 'bootstrap_iid_wcd', 'label': 'Bootstrap (I.I.D.)', 'color': 'red', 'marker': 'v'},
        {'col': 'bootstrap_ts_wcd', 'label': 'Bootstrap (Time-Series)', 'color': 'orange', 'marker': 'x'}
    ]
    plot_mc_metric_comparison(summary_df, wcd_configs, 'T', 'Mean WCD (log scale)', 'MC: Mean WCD Comparison', os.path.join(figures_dir, "mc_comparison_wcd.png"))

    # --- Plot configuration for Max Deviation in 'a' and 'b' ---
    max_dev_configs = [
        {'col': 'dd_bounds_max_dev_a', 'label': 'DD-Bounds (a)', 'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        {'col': 'bootstrap_iid_max_dev_a', 'label': 'Bootstrap IID (a)', 'color': 'red', 'marker': 'v', 'linestyle': '-'},
        {'col': 'bootstrap_ts_max_dev_a', 'label': 'Bootstrap TS (a)', 'color': 'orange', 'marker': 'x', 'linestyle': '-'},
        {'col': 'dd_bounds_max_dev_b', 'label': 'DD-Bounds (b)', 'color': 'blue', 'marker': 'o', 'linestyle': '--'},
        {'col': 'bootstrap_iid_max_dev_b', 'label': 'Bootstrap IID (b)', 'color': 'red', 'marker': 'v', 'linestyle': '--'},
        {'col': 'bootstrap_ts_max_dev_b', 'label': 'Bootstrap TS (b)', 'color': 'orange', 'marker': 'x', 'linestyle': '--'},
    ]
    plot_mc_metric_comparison(summary_df, max_dev_configs, 'T', 'Mean Max Deviation (log scale)', 'MC: Mean Max Deviation for "a" and "b"', os.path.join(figures_dir, "mc_comparison_dev_a_and_b.png"))

    print("\n--- Monte Carlo Simulation Finished Successfully! ---")