"""
WORKER SCRIPT to validate the empirical coverage probability of the three
implemented confidence region methods for a FIXED number of data points T.
"""

import os
import numpy as np
from tqdm import tqdm
from scipy.stats import chi2
from typing import Dict

# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_iid_samples, generate_time_series_data
from src.system_identification import estimate_least_squares_iid, estimate_least_squares_timeseries, perform_bootstrap_analysis
from src.set_membership import calculate_ellipse_from_qmi
from src.analysis import ConfidenceRectangle, ConfidenceEllipse, calculate_p_matrix_for_confidence_ellipse

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def perform_coverage_run(T: int, num_mc_runs: int = 1000) -> Dict[str, float]:
    """
    Orchestrates a Monte Carlo simulation to test the coverage probability for a specific T.

    Args:
        T (int): The number of data points (timesteps) to use for each run.
        num_mc_runs (int): The number of Monte Carlo iterations.

    Returns:
        A dictionary containing the calculated failure rates for each method.
    """
    print(f"--- Running Coverage Validation for T={T} with {num_mc_runs} runs ---")

    # === 3. Central Configuration ===
    TRUE_PARAMS_DICT = {'a': 0.5, 'b': 0.5}
    TRUE_PARAMS_TUPLE = tuple(TRUE_PARAMS_DICT.values())
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_ITERATIONS = 1000
    DEGREES_OF_FREEDOM = 2
    TUNING_FACTOR = 0.3
    # === 4. Initialize failure counters ===
    failure_counts = { 'dd_bounds': 0, 'bootstrap': 0, 'set_membership': 0 }

    # === 5. Main Monte Carlo Loop ===
    for i in tqdm(range(num_mc_runs), desc=f"Coverage Validation (T={T})", leave=False):
        
        # --- Pipeline 1: Data-Dependent Bounds on I.I.D. Data ---
        try:
            x_iid, u_iid, _, y_iid = generate_iid_samples(
                N=T, system_params=TRUE_PARAMS_DICT,
                params_config={'x_std_dev': 1.0, 'u_std_dev': INPUT_STD_DEV_U, 'w_std_dev': NOISE_STD_DEV_W},
                output_path=GENERATED_DATA_DIR, base_filename=f"temp_coverage_iid_run{i}", seed=i
            )
            A_est_dd, B_est_dd = estimate_least_squares_iid(x_iid, u_iid, y_iid)
            if A_est_dd is not None:
                p_matrix = calculate_p_matrix_for_confidence_ellipse(x_iid, u_iid, NOISE_STD_DEV_W, CONFIDENCE_DELTA, TUNING_FACTOR)
                ellipse_dd = ConfidenceEllipse(center=(A_est_dd.item(), B_est_dd.item()), p_matrix=p_matrix)
                if not ellipse_dd.contains(TRUE_PARAMS_TUPLE):
                    failure_counts['dd_bounds'] += 1
        except Exception:
            failure_counts['dd_bounds'] += 1

        # --- Generate shared time-series data for the next two pipelines ---
        state_ts, input_ts, _ = generate_time_series_data(
            system_params=TRUE_PARAMS_DICT, timesteps=T, 
            output_path=GENERATED_DATA_DIR, base_filename=f"temp_coverage_ts_run{i}",
            noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W}, seed=i
        )
        state_ts, input_ts = np.array([state_ts.flatten()]), np.array([input_ts.flatten()])

        # --- Pipeline 2: Bootstrap Dean on Time-Series Data ---
        try:
            A_est_bs, B_est_bs = estimate_least_squares_timeseries(state_ts, input_ts)
            bootstrap_results = perform_bootstrap_analysis(
                initial_estimate=(A_est_bs, B_est_bs), data_shape=(1, T),
                sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W}, M=BOOTSTRAP_ITERATIONS,
                delta=CONFIDENCE_DELTA, seed=i + 1
            )
            rect = ConfidenceRectangle(center=(A_est_bs.item(), B_est_bs.item()), epsilons=(bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B']))
            if not rect.contains(TRUE_PARAMS_TUPLE):
                failure_counts['bootstrap'] += 1
        except Exception:
            failure_counts['bootstrap'] += 1

        # --- Pipeline 3: Set Membership via direct QMI on Time-Series Data ---
        try:
            X_plus, X_minus, U_minus = state_ts[:, 1:T+1], state_ts[:, :T], input_ts[:, :T]
            c_delta = chi2.ppf(1 - CONFIDENCE_DELTA, df=DEGREES_OF_FREEDOM)
            Phi11 = (NOISE_STD_DEV_W**2) * c_delta * np.eye(1); Phi12 = np.zeros((1, T)); Phi21 = Phi12.T
            Z_reg = np.vstack([X_minus, U_minus])
            Phi22 = -np.linalg.pinv(Z_reg) @ Z_reg
            qmi_results = calculate_ellipse_from_qmi(X_plus, X_minus, U_minus, Phi11, Phi12, Phi21, Phi22)
            if qmi_results:
                ellipse_qmi = ConfidenceEllipse(center=qmi_results['center'], p_matrix=qmi_results['shape_matrix'])
                if not ellipse_qmi.contains(TRUE_PARAMS_TUPLE):
                    failure_counts['set_membership'] += 1
            else:
                failure_counts['set_membership'] += 1
        except Exception:
            failure_counts['set_membership'] += 1

    # === 6. Calculate and return failure rates ===
    failure_rates = {
        f"{method}_failure_rate": count / num_mc_runs
        for method, count in failure_counts.items()
    }
    
    return failure_rates

# This block allows the script to be run directly for a single test run
if __name__ == '__main__':
    # Perform a single run with a default T and number of runs
    results = perform_coverage_run(T=100, num_mc_runs=1000)

    # Display the results in a formatted table
    print("\n--- Coverage Validation Results ---")
    print("="*45)
    print(f"{'Method':<25} | {'Failure Rate':>15}")
    print("-"*45)
    for method_key, rate in results.items():
        method_name = method_key.replace('_failure_rate', '')
        print(f"{method_name:<25} | {rate:>14.2%}")
    print("="*45)




def perform_bootstrap_only_coverage_run(T: int, num_mc_runs: int = 1000, seed_base: int = 0) -> Dict[str, float]:
    """
    WORKER: Performs a Monte Carlo run to find the failure rate ONLY for the Bootstrap method.
    """
    # --- Configuration for this specific run ---
    TRUE_PARAMS_DICT = {'a': 0.5, 'b': 0.5}
    TRUE_PARAMS_TUPLE = tuple(TRUE_PARAMS_DICT.values())
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_ITERATIONS = 2000
    
    failure_count = 0

    # --- Main Monte Carlo Loop ---
    for i in range(num_mc_runs):
        current_seed = seed_base * 10 + i

        # --- Generate time-series data for the bootstrap pipeline ---
        state_ts, input_ts, _ = generate_time_series_data(
            system_params=TRUE_PARAMS_DICT, timesteps=T, 
            output_path=GENERATED_DATA_DIR,                 
            base_filename=f"temp_coverage_variance_run{i}",
            noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W}, 
            seed=current_seed
        )
        
        state_ts, input_ts = np.array([state_ts.flatten()]), np.array([input_ts.flatten()])

        # --- Run Bootstrap and check coverage ---
        try:
            A_est_bs, B_est_bs = estimate_least_squares_timeseries(state_ts, input_ts)
            bootstrap_results = perform_bootstrap_analysis(
                initial_estimate=(A_est_bs, B_est_bs), data_shape=(1, T),
                sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W}, M=BOOTSTRAP_ITERATIONS,
                delta=(CONFIDENCE_DELTA/2), seed=current_seed + 1
            )
            rect = ConfidenceRectangle(center=(A_est_bs.item(), B_est_bs.item()), epsilons=(bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B']))
            if not rect.contains(TRUE_PARAMS_TUPLE):
                failure_count += 1
        except Exception:
            failure_count += 1
    
    return {'bootstrap_failure_rate': failure_count / num_mc_runs}

def perform_set_membership_only_coverage_run(T: int, num_mc_runs: int = 1000, seed_base: int = 0) -> dict:
    """
    WORKER: Performs a Monte Carlo run to find the failure rate ONLY for the Set Membership method.
    This version is corrected to exactly match the logic of the original, working Pipeline 3.
    """
    # --- Configuration for this specific run ---
    TRUE_PARAMS_DICT = {'a': 0.5, 'b': 0.5}
    TRUE_PARAMS_TUPLE = tuple(TRUE_PARAMS_DICT.values())
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    CONFIDENCE_DELTA = 0.05
    DEGREES_OF_FREEDOM = 2
    
    # --- Define project paths locally for the worker ---
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
    
    failure_count = 0

    # --- Main Monte Carlo Loop ---
    for i in range(num_mc_runs):
        current_seed = seed_base * 10 + i

        # --- Generate time-series data ---
        state_ts, input_ts, _ = generate_time_series_data(
            system_params=TRUE_PARAMS_DICT, 
            timesteps=T, 
            output_path=GENERATED_DATA_DIR,
            base_filename=f"temp_set_membership_run_{i}",
            noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W}, 
            seed=current_seed
        )
        state_ts, input_ts = np.array([state_ts.flatten()]), np.array([input_ts.flatten()])

        # --- Run Set Membership and check coverage ---
        try:
            # KORREKTUR: Das Slicing muss exakt wie in der funktionierenden Pipeline 3 sein.
            X_plus, X_minus, U_minus = state_ts, state_ts, input_ts
            
            c_delta = chi2.ppf(1 - CONFIDENCE_DELTA, df=DEGREES_OF_FREEDOM)
            Phi11 = (NOISE_STD_DEV_W**2) * c_delta * np.eye(1); Phi12 = np.zeros((1, T)); Phi21 = Phi12.T
            Z_reg = np.vstack([X_minus, U_minus])
            Phi22 = -np.linalg.pinv(Z_reg) @ Z_reg
            
            qmi_results = calculate_ellipse_from_qmi(X_plus, X_minus, U_minus, Phi11, Phi12, Phi21, Phi22)
            
            if qmi_results:
                ellipse_qmi = ConfidenceEllipse(center=qmi_results['center'], p_matrix=qmi_results['shape_matrix'])
                if not ellipse_qmi.contains(TRUE_PARAMS_TUPLE):
                    failure_count += 1
            else:
                failure_count += 1 # Zählt als Fehler, wenn QMI nicht lösbar ist
        except Exception:
            failure_count += 1
            
    return {'set_membership_failure_rate': failure_count / num_mc_runs}