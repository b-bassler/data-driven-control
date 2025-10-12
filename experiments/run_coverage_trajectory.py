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
from src.data_generation import generate_trajectory_data
from src.system_identification import estimate_least_squares_trajectory, perform_bootstrap_analysis_trajectory
from src.set_membership import calculate_ellipse_from_qmi
from src.analysis import ConfidenceRectangle, ConfidenceEllipse, calculate_p_matrix_trajectory

def perform_coverage_run_trajectory(T: int, num_mc_runs: int = 10) -> Dict[str, float]:
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
    TRUE_PARAMS = {'a': 0.99, 'b': 0.5}
    TRUE_PARAMS_TUPLE = tuple(TRUE_PARAMS.values())
    NOISE_STD_DEV_W = 0.1
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_ITERATIONS = 2000
    DEGREES_OF_FREEDOM = 2
      # Parameters specific to the Tsiams method
    C = 0.00140625
    TAU = 2


    # === 4. Initialize failure counters ===
    failure_counts = { 'dd_bounds': 0, 'bootstrap': 0, 'set_membership': 0 }

    # === 5. Main Monte Carlo Loop ===
    for i in tqdm(range(num_mc_runs), desc=f"Coverage Validation (T={T})", leave=False):
        
                # --- Generate shared time-series data for the upcoming pipelines ---
        state_ts, input_ts, _ = generate_trajectory_data(
            system_params=TRUE_PARAMS, timesteps=T, 
            output_path=GENERATED_DATA_DIR, base_filename=f"temp_coverage_ts_run{i}",
            noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W}, seed=i
        )


        #  shared OLS for Pipeline 1 and 2 
        A_est, B_est = estimate_least_squares_trajectory(state_ts, input_ts)


        # --- Pipeline 1: Data-Dependent Bounds on trajectory data ---
        try:
            if A_est is not None:
                tsiams_results = calculate_p_matrix_trajectory(
                    state_data=state_ts[:, :T], 
                    input_data=input_ts,
                    true_A=np.array([[TRUE_PARAMS['a']]]), 
                    true_B=np.array([[TRUE_PARAMS['b']]]),
                    sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W},
                    delta=CONFIDENCE_DELTA, 
                    c=C, 
                    tau=TAU
                )
                p_matrix = tsiams_results['p_matrix']
                ellipse_dd = ConfidenceEllipse(center=(A_est.item(), B_est.item()), p_matrix=p_matrix)
                if not ellipse_dd.contains(TRUE_PARAMS_TUPLE):
                    failure_counts['dd_bounds'] += 1
        except Exception:
            failure_counts['dd_bounds'] += 1


        # --- Pipeline 2: Bootstrap Dean on Time-Series Data ---
        try:
            bootstrap_results = perform_bootstrap_analysis_trajectory(
                initial_estimate=(A_est, B_est), data_shape=(1, T),
                sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W}, M=BOOTSTRAP_ITERATIONS,
                delta=(CONFIDENCE_DELTA/2), seed=i + 1
            )
            rect = ConfidenceRectangle(center=(A_est.item(), B_est.item()), epsilons=(bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B']))
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
    results = perform_coverage_run_trajectory(T=100, num_mc_runs=100)

    # Display the results in a formatted table
    print("\n--- Coverage Validation Results ---")
    print("="*45)
    print(f"{'Method':<25} | {'Failure Rate':>15}")
    print("-"*45)
    for method_key, rate in results.items():
        method_name = method_key.replace('_failure_rate', '')
        print(f"{method_name:<25} | {rate:>14.2%}")
    print("="*45)









def run_coverage_trajectory_over_T():
    """Performs the coverage validation for a range of T."""
    print("--- Starting Coverage Validation over a range of T ---")
    
    # Configuration 
    T_RANGE = [8, 10, 15, 20, 30, 40, 50, 70, 90, 110, 150, 200, 300, 400, 500]
    NUM_MC_RUNS_PER_T = 100 # Number of MC runs for each T-value
    CONFIDENCE_DELTA = 0.05 #for plot only, delta config in run_coverage_validation.py

    #Loop over T 
    results_list = []
    for T in tqdm(T_RANGE, desc="Total Progress"):
        # Run the full MC simulation for the current T
        single_result = perform_coverage_run_trajectory(T=T, num_mc_runs=NUM_MC_RUNS_PER_T)
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