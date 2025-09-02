"""
Experiment to validate the bootstrap coverage probability on I.I.D. data.

This script analyzes the bootstrap performance by varying the number of
I.I.D. samples (N) and checking the empirical coverage rate for parameters
A, B, and the joint region.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict

# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_iid_samples
from src.system_identification import estimate_least_squares_iid, perform_bootstrap_analysis_iid
from src.analysis import ConfidenceRectangle
from src.plotting import plot_bootstrap_coverage_trend

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_bootstrap_iid_validation_experiment():
    """Orchestrates the Bootstrap on I.I.D. data validation experiment."""
    print("--- Starting Bootstrap Coverage Validation (on I.I.D. Data) ---")

    # === 3. Central Configuration ===
    N_SAMPLES_RANGE = np.arange(20, 10021, 2000)
    NUM_MC_RUNS = 500
    
    TRUE_PARAMS_DICT = {'a': 0.5, 'b': 0.5}
    TRUE_PARAMS_TUPLE = tuple(TRUE_PARAMS_DICT.values())
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_ITERATIONS = 2000

    # === 4. Main Loop over the number of samples N ===
    results_list = []
    print(f"\nRunning validation for N in {N_SAMPLES_RANGE}...")
    for N in tqdm(N_SAMPLES_RANGE, desc="N Progress"):
        failure_counts = {'a': 0, 'b': 0, 'both': 0}

        for i in range(NUM_MC_RUNS):
            try:
                # --- CORRECTED DATA GENERATION CALL ---
                x_iid, u_iid, _, y_iid = generate_iid_samples(
                    N=N, system_params=TRUE_PARAMS_DICT,
                    params_config={'x_std_dev': 1.0, 'u_std_dev': INPUT_STD_DEV_U, 'w_std_dev': NOISE_STD_DEV_W},
                    output_path=GENERATED_DATA_DIR,
                    base_filename=f"temp_bootstrap_iid_val_run{i}",
                    seed=i
                )
                
                # --- Run Bootstrap and check coverage ---
                A_est, B_est = estimate_least_squares_iid(x_iid, u_iid, y_iid)
                if A_est is None:
                    failure_counts['a'] += 1; failure_counts['b'] += 1; failure_counts['both'] += 1
                    continue

                bootstrap_results = perform_bootstrap_analysis_iid(
                    initial_estimate=(A_est, B_est), N=N,
                    sigmas={'x': 1.0, 'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W}, M=BOOTSTRAP_ITERATIONS,
                    delta=CONFIDENCE_DELTA, seed=i + 1
                )
                rect = ConfidenceRectangle(center=(A_est.item(), B_est.item()), epsilons=(bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B']))
                
                coverage = rect.contains_per_parameter(TRUE_PARAMS_TUPLE)
                if not coverage['a']: failure_counts['a'] += 1
                if not coverage['b']: failure_counts['b'] += 1
                if not coverage['both']: failure_counts['both'] += 1
            except Exception:
                failure_counts['a'] += 1; failure_counts['b'] += 1; failure_counts['both'] += 1
        
        results_list.append({
            'N': N,
            'failure_rate_a': failure_counts['a'] / NUM_MC_RUNS,
            'failure_rate_b': failure_counts['b'] / NUM_MC_RUNS,
            'failure_rate_both': failure_counts['both'] / NUM_MC_RUNS,
        })

    # === 5. Process, save, and plot results ===
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(RESULTS_DIR, "bootstrap_validation_iid_vs_N.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n-> Full validation results saved to {results_path}")

    figures_dir = os.path.join(RESULTS_DIR, "figures", "bootstrap_validation")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "coverage_iid_vs_N.png")
    plot_bootstrap_coverage_trend(results_df, target_rate=CONFIDENCE_DELTA, output_path=plot_path, x_axis_label="Number of I.I.D. Samples (N)")

    print("\n--- Bootstrap I.I.D. Validation Experiment Finished ---")