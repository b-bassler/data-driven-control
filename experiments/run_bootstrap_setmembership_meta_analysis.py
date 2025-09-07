"""
META-EXPERIMENT SCRIPT
Orchestrates multiple runs of the coverage validation for multiple methods
to calculate mean and standard deviation of the failure rates.
This version is parallelized to leverage multiple CPU cores.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# --- 1. Import required modules ---
# Import BOTH worker functions now
from experiments.run_coverage_validation import perform_bootstrap_only_coverage_run, perform_set_membership_only_coverage_run
# We will create a new, more general plotting function
from src.plotting import plot_coverage_meta_comparison 

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# --- Helper functions for parallel execution ---
def run_single_bootstrap_run(seed_base: int, T: int, num_mc_runs: int) -> float:
    """Helper for parallel execution of a single Bootstrap meta-run."""
    result = perform_bootstrap_only_coverage_run(T=T, num_mc_runs=num_mc_runs, seed_base=seed_base)
    return result['bootstrap_failure_rate']

def run_single_set_membership_run(seed_base: int, T: int, num_mc_runs: int) -> float:
    """Helper for parallel execution of a single Set Membership meta-run."""
    result = perform_set_membership_only_coverage_run(T=T, num_mc_runs=num_mc_runs, seed_base=seed_base)
    return result['set_membership_failure_rate']

def run_coverage_meta_experiment(T_values: list[int], num_meta_runs: int, num_mc_runs: int) -> pd.DataFrame:
    """
    Executes the meta-experiment in parallel for multiple methods.
    """
    all_results = []
    for T in tqdm(T_values, desc="Overall Progress (T-Values)"):
        
        meta_run_indices = range(num_meta_runs)

        # --- Run Bootstrap Simulations in Parallel ---
        bootstrap_task = partial(run_single_bootstrap_run, T=T, num_mc_runs=num_mc_runs)
        bootstrap_rates = []
        with ProcessPoolExecutor() as executor:
            results_iterator = executor.map(bootstrap_task, meta_run_indices)
            for result in tqdm(results_iterator, total=num_meta_runs, desc=f"Bootstrap Meta-Runs (T={T})", leave=False):
                bootstrap_rates.append(result)

        # --- Run Set Membership Simulations in Parallel ---
        set_membership_task = partial(run_single_set_membership_run, T=T, num_mc_runs=num_mc_runs)
        set_membership_rates = []
        with ProcessPoolExecutor() as executor:
            results_iterator = executor.map(set_membership_task, meta_run_indices)
            for result in tqdm(results_iterator, total=num_meta_runs, desc=f"Set Membership Meta-Runs (T={T})", leave=False):
                set_membership_rates.append(result)

        # --- Calculate and store statistics for BOTH methods ---
        all_results.append({
            'T': T,
            'bootstrap_mean_rate': np.mean(bootstrap_rates),
            'bootstrap_std_dev_rate': np.std(bootstrap_rates),
            'set_membership_mean_rate': np.mean(set_membership_rates),
            'set_membership_std_dev_rate': np.std(set_membership_rates),
        })
        
    return pd.DataFrame(all_results)

def run_bootstrap_setmembership_meta_analysis_experiment():
    """Main entry point for the meta-analysis experiment."""
    # --- Configuration for the Meta-Experiment ---
    T_VALUES_TO_TEST = [20, 30]
    NUM_META_RUNS = 5
    NUM_MC_RUNS_PER_EXPERIMENT = 50
    TARGET_FAILURE_RATE = 0.05
    
    print("--- Starting PARALLEL Meta-Experiment for Coverage Validation ---")

    start_time = time.time()
    final_results_df = run_coverage_meta_experiment(
        T_values=T_VALUES_TO_TEST,
        num_meta_runs=NUM_META_RUNS,
        num_mc_runs=NUM_MC_RUNS_PER_EXPERIMENT
    )
    end_time = time.time()
    print(f"\nMeta-experiment completed. Total duration: {end_time - start_time:.2f} seconds.")

    # --- Save and Plot the results ---
    results_path = os.path.join(RESULTS_DIR, "coverage_meta_analysis_summary.csv")
    final_results_df.to_csv(results_path, index=False)
    print(f"-> Meta-analysis summary saved to {results_path}")

    plot_dir = os.path.join(RESULTS_DIR, "figures", "meta_analysis")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'coverage_meta_comparison.png')
    
    # Define plot configurations for the new plotting function
    plot_configs = [
        {
            'mean_col': 'bootstrap_mean_rate',
            'std_dev_col': 'bootstrap_std_dev_rate',
            'label': 'Bootstrap',
            'color': 'orange',
            'marker': 'x'
        },
        {
            'mean_col': 'set_membership_mean_rate',
            'std_dev_col': 'set_membership_std_dev_rate',
            'label': 'Set Membership (QMI)',
            'color': 'green',
            'marker': 's'
        }
    ]
    
    plot_coverage_meta_comparison(
        dataframe=final_results_df,
        plot_configs=plot_configs,
        target_rate=TARGET_FAILURE_RATE,
        output_path=plot_path
    )

if __name__ == '__main__':
    run_bootstrap_setmembership_meta_analysis_experiment()