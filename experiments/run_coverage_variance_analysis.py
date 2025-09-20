"""
Orchestrates a "meta" Monte Carlo simulation to analyze the variance
of the empirical coverage rate of the Bootstrap method.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Add project root to path to allow imports from src and other experiments
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the "worker" function from the other experiment script and the plotter
from old.run_coverage_validation import perform_bootstrap_only_coverage_run
from src.plotting import plot_histogram

# Define project paths
RESULTS_DIR = os.path.join(project_root, 'results')

def run_coverage_variance_analysis():
    """
    Performs the outer Monte Carlo loop to get a distribution of failure rates for Bootstrap.
    """
    # === Configuration ===
    NUM_OUTER_RUNS = 30  # How many times to repeat the entire coverage experiment
    NUM_INNER_RUNS = 1000 # How many MC runs within each coverage experiment
    T_DATA_POINTS = 100  # The fixed T for which we analyze the variance

    print(f"--- Starting Variance Analysis of Bootstrap Coverage ---")
    print(f"Performing {NUM_OUTER_RUNS} outer runs, each with {NUM_INNER_RUNS} inner simulations at T={T_DATA_POINTS}.")

    bootstrap_failure_rates = []

    for i in tqdm(range(NUM_OUTER_RUNS), desc="Outer MC Loop (Variance Analysis)"):
        # Call the lean worker function from the other script
        single_run_results = perform_bootstrap_only_coverage_run(
            T=T_DATA_POINTS, 
            num_mc_runs=NUM_INNER_RUNS,
            seed_base=i # Pass the outer loop index as a base for the seed
        )
        
        if 'bootstrap_failure_rate' in single_run_results:
            bootstrap_failure_rates.append(single_run_results['bootstrap_failure_rate'])
    
    # --- Analyze the collected distribution of failure rates ---
    if not bootstrap_failure_rates:
        print("Warning: No results collected. Aborting analysis.")
        return

    rates_array = np.array(bootstrap_failure_rates)
    
    mean_rate = np.mean(rates_array)
    std_dev_rate = np.std(rates_array)
    conf_interval = np.percentile(rates_array, [2.5, 97.5])
    
    print("\n--- Analysis of Failure Rate Distribution ---")
    print(f"  - Number of full experiments: {len(rates_array)}")
    print(f"  - Mean Failure Rate:          {mean_rate:.2%}")
    print(f"  - Std. Dev. of Failure Rate:  {std_dev_rate:.2%}")
    print(f"  - 95% Confidence Interval for Rate:  [{conf_interval[0]:.2%}, {conf_interval[1]:.2%}]")

    # --- Save and plot the results ---
    results_df = pd.DataFrame({'bootstrap_failure_rate': rates_array})
    results_path = os.path.join(RESULTS_DIR, "coverage_variance_bootstrap.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n-> Distribution of failure rates saved to {results_path}")

    figures_dir = os.path.join(RESULTS_DIR, "figures", "coverage_validation")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "histogram_bootstrap_coverage.png")
    
    plot_histogram(rates_array, 
                   title=f'Distribution of Bootstrap Failure Rate (T={T_DATA_POINTS})',
                   x_label='Estimated Failure Rate',
                   output_path=plot_path)

    print("\n--- Coverage Variance Analysis Finished Successfully! ---")

# Guard to allow direct execution for testing purposes
if __name__ == '__main__':
    run_coverage_variance_analysis()