import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import chi2
from typing import Dict
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)



RESULTS_DIR = os.path.join(project_root, 'results')
GENERATED_DATA_DIR = os.path.join(project_root, 'data', 'generated')


from src.plotting import plot_coverage_trend
from src.data_generation import generate_iid_samples
from src.system_identification import estimate_least_squares_iid, perform_bootstrap_analysis_iid
from src.set_membership import calculate_ellipse_from_qmi
from src.analysis import ConfidenceRectangle, ConfidenceEllipse, calculate_p_matrix_ddbounds_iid



def perform_coverage_run_iid(N: int, num_mc_runs: int = 1000) -> Dict[str, float]:
    """
    Orchestrates a Monte Carlo simulation to test the coverage probability
    of different system identification methods on I.I.D. data.

    Args:
        N: The number of data points (samples) to use for each run.
        num_mc_runs: The number of Monte Carlo iterations.

    Returns:
        A dictionary containing the calculated failure rates for each method.
    """
    print(f"--- Running Coverage Validation for I.I.D. Data (N={N}, Runs={num_mc_runs}) ---")

    # 1. Central Configuration
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    TRUE_PARAMS_TUPLE = tuple(TRUE_PARAMS.values())
    NOISE_STD_DEV_W = 0.1
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_ITERATIONS = 2000
    DEGREES_OF_FREEDOM = 2 # Note: For QMI, based on the number of estimated parameters.
    TUNING_FACTOR = 1

    # 2. Initialize failure counters
    failure_counts = {'dd_bounds': 0, 'bootstrap': 0, 'set_membership': 0}

    # 3. Main Monte Carlo Loop
    for i in tqdm(range(num_mc_runs), desc=f"Coverage Validation (N={N})", leave=False):
        
        # --- Generate I.I.D. data ---
        # Data is now generated in the correct (features, samples) format, e.g., (1, N).
        x_iid, u_iid, _, y_iid = generate_iid_samples(
            N=N, system_params=TRUE_PARAMS,
            params_config={'x_std_dev': 1.0, 'u_std_dev': INPUT_STD_DEV_U, 'w_std_dev': NOISE_STD_DEV_W},
            output_path=GENERATED_DATA_DIR, base_filename=f"temp_coverage_iid_run{i}", seed=i
        )

        # --- Perform initial OLS estimation ---
        A_est, B_est = estimate_least_squares_iid(x_iid, u_iid, y_iid)
        if A_est is None: continue # Skip iteration if estimation fails

        # --- Pipeline 1: Data-Dependent Bounds ---
        # is compatible with the new (1, N) data format.
        try:
            p_matrix = calculate_p_matrix_ddbounds_iid(x_iid, u_iid, NOISE_STD_DEV_W, CONFIDENCE_DELTA, TUNING_FACTOR)
            ellipse_dd = ConfidenceEllipse(center=(A_est.item(), B_est.item()), p_matrix=p_matrix)
            if not ellipse_dd.contains(TRUE_PARAMS_TUPLE):
                failure_counts['dd_bounds'] += 1
        except Exception:
            failure_counts['dd_bounds'] += 1

        # --- Pipeline 2: Parametric Bootstrap ---
        try:
            bootstrap_results = perform_bootstrap_analysis_iid(
                initial_estimate=(A_est, B_est),
                N=N, # Pass N directly
                sigmas={'x': 1.0, 'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W},
                M=BOOTSTRAP_ITERATIONS,
                delta=CONFIDENCE_DELTA / 2, # Using delta/2 for a two-sided interval
                seed=i + 1
            )
            rect = ConfidenceRectangle(center=(A_est.item(), B_est.item()), epsilons=(bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B']))
            if not rect.contains(TRUE_PARAMS_TUPLE):
                failure_counts['bootstrap'] += 1
        except Exception:
            failure_counts['bootstrap'] += 1


        # --- Pipeline 3: Set Membership (QMI) ---
        try:
            X_plus, X_minus, U_minus = y_iid, x_iid, u_iid
            
            c_delta = chi2.ppf(1 - CONFIDENCE_DELTA, df=DEGREES_OF_FREEDOM)
            Phi11 = (NOISE_STD_DEV_W**2) * c_delta * np.eye(1)
            Phi12 = np.zeros((1, N)) # CHANGED: Use N instead of T
            Phi21 = Phi12.T
            
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

    # 4. Calculate and return failure rates
    failure_rates = {
        f"{method}_failure_rate": count / num_mc_runs
        for method, count in failure_counts.items()
    }
    
    return failure_rates




# This block allows the script to be run directly for a single test run
if __name__ == '__main__':
    # Perform a single run with a default T and number of runs
    results = perform_coverage_run_iid(N=100, num_mc_runs=10)

    # Display the results in a formatted table
    print("\n--- Coverage Validation Results ---")
    print("="*45)
    print(f"{'Method':<25} | {'Failure Rate':>15}")
    print("-"*45)
    for method_key, rate in results.items():
        method_name = method_key.replace('_failure_rate', '')
        print(f"{method_name:<25} | {rate:>14.2%}")
    print("="*45)













def run_coverage_iid_over_T():
    """Performs the coverage validation for a range of T."""
    print("--- Starting Coverage Validation over a range of T ---")
    
    # Configuration 
    T_RANGE = [ 10, 15, 20, 30, 40, 50, 70, 90, 110, 150, 200, 300, 400, 500]
    NUM_MC_RUNS_PER_T = 800 # Number of MC runs for each T-value
    CONFIDENCE_DELTA = 0.05 #for plot only, delta config in run_coverage_validation.py

    #Loop over T 
    results_list = []
    for T in tqdm(T_RANGE, desc="Total Progress"):
        # Run the full MC simulation for the current T
        single_result = perform_coverage_run_iid(N=T, num_mc_runs=NUM_MC_RUNS_PER_T)
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