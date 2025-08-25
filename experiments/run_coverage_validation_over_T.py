

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from experiments.run_coverage_validation import perform_coverage_run
from src.plotting import plot_coverage_trend

# Define project paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_coverage_validation_over_T():
    """Performs the coverage validation for a range of T."""
    print("--- Starting Coverage Validation over a range of T ---")
    
    # Configuration 
    T_RANGE = np.arange(10, 300, 20)
    NUM_MC_RUNS_PER_T = 100 # Number of MC runs for each T-value
    CONFIDENCE_DELTA = 0.05 #for plot only, delta config in run_coverage_validation.py

    #Loop over T 
    results_list = []
    for T in tqdm(T_RANGE, desc="Total Progress"):
        # Run the full MC simulation for the current T
        single_result = perform_coverage_run(T=T, num_mc_runs=NUM_MC_RUNS_PER_T)
        # Add the current T to the dictionary for our DataFrame
        single_result['T'] = T
        results_list.append(single_result)
        
    # Process and save results 
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(RESULTS_DIR, "coverage_validation_over_T.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n-> Full coverage results saved to {results_path}")

    # Generate the final plot 
    figures_dir = os.path.join(RESULTS_DIR, "figures", "coverage_validation")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "coverage_trend.png")
    plot_coverage_trend(results_df, target_rate=CONFIDENCE_DELTA, output_path=plot_path)

    print("\n--- Coverage Validation over T Finished ---")