"""
Experiment script to empirically calibrate the constant C of the 
Data-Dependent Bounds method by analyzing the failure rate over a
range of tuning factors.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict

# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_iid_samples
from src.system_identification import estimate_least_squares_iid
from src.analysis import ConfidenceEllipse, calculate_p_matrix_ddbounds_iid
from src.plotting import plot_calibration_curve # Import is now active

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def perform_dd_bounds_coverage_run(T: int, num_mc_runs: int, tuning_factor: float) -> float:
    """
    WORKER: Performs a Monte Carlo run to find the failure rate for a specific
    T and a specific tuning_factor.
    """
    # --- Configuration for this specific run ---
    TRUE_PARAMS_DICT = {'a': 0.5, 'b': 0.5}
    TRUE_PARAMS_TUPLE = tuple(TRUE_PARAMS_DICT.values())
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    
    failure_count = 0

    # --- Main Monte Carlo Loop ---
    for i in range(num_mc_runs):
        try:
            x_iid, u_iid, _, y_iid = generate_iid_samples(
                N=T, system_params=TRUE_PARAMS_DICT,
                params_config={'x_std_dev': 1.0, 'u_std_dev': INPUT_STD_DEV_U, 'w_std_dev': NOISE_STD_DEV_W},
                output_path=GENERATED_DATA_DIR, # Provide a valid path
                base_filename=f"temp_calib_run_{i}", # Provide a valid filename
                seed=i
            )
            A_est, B_est = estimate_least_squares_iid(x_iid, u_iid, y_iid)
            if A_est is not None:
                # Use the tuning_factor when calculating the P-matrix
                p_matrix = calculate_p_matrix_ddbounds_iid(
                    x_iid, u_iid, NOISE_STD_DEV_W, CONFIDENCE_DELTA,
                    tuning_factor=tuning_factor
                )
                ellipse = ConfidenceEllipse(center=(A_est.item(), B_est.item()), p_matrix=p_matrix)
                if not ellipse.contains(TRUE_PARAMS_TUPLE):
                    failure_count += 1
        except Exception:
            failure_count += 1
            
    return failure_count / num_mc_runs


def run_dd_bounds_calibration_experiment():
    """
    MANAGER: Orchestrates the calibration experiment by looping over tuning factors.
    """
    print("--- Starting DD-Bounds Calibration Experiment ---")

    # === Central Configuration ===
    T_DATA_POINTS = 400
    NUM_MC_RUNS = 1000
    CONFIDENCE_DELTA = 0.05
    # A wider range to better see the curve's behavior
    TUNING_FACTOR_RANGE = np.linspace(0.1, 0.8, 30) 

    # === Loop over tuning factors and collect failure rates ===
    results_list = []
    print(f"\nRunning calibration for {len(TUNING_FACTOR_RANGE)} tuning factors...")
    for factor in tqdm(TUNING_FACTOR_RANGE, desc="Calibration Progress"):
        # Call the worker function for the current factor
        failure_rate = perform_dd_bounds_coverage_run(
            T=T_DATA_POINTS,
            num_mc_runs=NUM_MC_RUNS,
            tuning_factor=factor
        )
        results_list.append({
            'tuning_factor': factor,
            'failure_rate': failure_rate
        })

    # === Process, save, and plot the final results ===
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(RESULTS_DIR, "dd_bounds_calibration.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n-> Full calibration data saved to {results_path}")

    # === Generate the calibration plot ===
    print("\nGenerating calibration plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "calibration")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "dd_bounds_calibration_curve.png")
    
    # The placeholder is now replaced with the actual function call
    plot_calibration_curve(results_df, target_rate=CONFIDENCE_DELTA, output_path=plot_path)

    print("\n--- DD-Bounds Calibration Finished Successfully! ---")