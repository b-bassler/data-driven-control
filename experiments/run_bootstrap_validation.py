"""
Experiment to validate the bootstrap coverage probability by varying the
number of rollouts (N) and trajectory length (T) while keeping the
total number of data points (N*T) constant.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict

# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_trajectory_data
from src.system_identification import estimate_least_squares_trajectory, perform_bootstrap_analysis_trajectory
from src.analysis import ConfidenceRectangle
from src.plotting import plot_bootstrap_coverage_trend

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated') # Path for temporary data
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_bootstrap_validation_experiment():
    """Orchestrates the Bootstrap validation experiment."""
    print("--- Starting Bootstrap Coverage Validation (N vs. T) ---")

    # === 3. Central Configuration ===
    TOTAL_DATA_POINTS = 5000
    N_ROLLOUTS_RANGE = [10, 25, 50, 100, 250, 500 ,1000,1250,] 
    NUM_MC_RUNS = 100
    
    TRUE_PARAMS_DICT = {'a': 0.5, 'b': 0.5}
    TRUE_PARAMS_TUPLE = tuple(TRUE_PARAMS_DICT.values())
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_ITERATIONS = 500

    # === 4. Main Loop over (N, T) combinations ===
    results_list = []
    print(f"\nRunning validation for N in {N_ROLLOUTS_RANGE} with total data {TOTAL_DATA_POINTS}...")
    for N in tqdm(N_ROLLOUTS_RANGE, desc="N Progress"):
        T = TOTAL_DATA_POINTS // N
        if T == 0: continue

        failure_counts = {'a': 0, 'b': 0, 'both': 0}

        for i in range(NUM_MC_RUNS):
            # --- Generate N rollouts of length T ---
            all_state_ts, all_input_ts = [], []
            for n in range(N):
                state_ts_raw, input_ts_raw, _ = generate_trajectory_data(
                    system_params=TRUE_PARAMS_DICT, timesteps=T,
                    output_path=GENERATED_DATA_DIR,
                    base_filename=f"temp_bootstrap_val_run{i}_rollout{n}",
                    noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W},
                    seed=(i * 1000) + n
                )
                all_state_ts.append(state_ts_raw.flatten())
                all_input_ts.append(input_ts_raw.flatten())
            state_ts = np.array(all_state_ts)
            input_ts = np.array(all_input_ts)

            # --- Run Bootstrap and check coverage ---
            try:
                A_est, B_est = estimate_least_squares_trajectory(state_ts, input_ts)
                bootstrap_results = perform_bootstrap_analysis_trajectory(
                    initial_estimate=(A_est, B_est), data_shape=(N, T),
                    sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W}, M=BOOTSTRAP_ITERATIONS,
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
            'N': N, 'T': T,
            'failure_rate_a': failure_counts['a'] / NUM_MC_RUNS,
            'failure_rate_b': failure_counts['b'] / NUM_MC_RUNS,
            'failure_rate_both': failure_counts['both'] / NUM_MC_RUNS,
        })

    # === 5. Process, save, and plot results ===
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(RESULTS_DIR, "bootstrap_validation_vs_N.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n-> Full validation results saved to {results_path}")

    figures_dir = os.path.join(RESULTS_DIR, "figures", "bootstrap_validation")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "coverage_vs_N.png")
    plot_bootstrap_coverage_trend(results_df, target_rate=CONFIDENCE_DELTA, output_path=plot_path)

    print("\n--- Bootstrap Validation Experiment Finished ---")