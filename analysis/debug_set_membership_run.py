# analysis/debug_set_membership_run.py

import os
import sys
import numpy as np
from scipy.stats import chi2

# --- Add project root to Python path to allow imports from src ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# --- Import all required tools from the src library ---
from src.data_generation import generate_time_series_data
from src.set_membership import calculate_ellipse_from_qmi
from src.analysis import ConfidenceEllipse
from src.plotting import plot_qmi_ellipse

# ======================================================================
# === DEBUGGING CONFIGURATION ===
T_TO_DEBUG = 70
SEED_TO_DEBUG = 80
# ======================================================================

# --- Define project paths ---
RESULTS_DIR = os.path.join(project_root, 'results')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')



def debug_single_run(T: int, data_seed: int):
    """
    Orchestrates a single, detailed run of the Set Membership pipeline for debugging.
    """
    print(f"--- Starting Debug Run for T={T}, Seed={data_seed} ---")

    # === 1. Central Configuration (copied from the main experiment) ===
    TRUE_PARAMS = {'a': 0.99, 'b': 0.5}
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    CONFIDENCE_DELTA = 0.05
    DEGREES_OF_FREEDOM = 2

    # === 2. Generate the exact same dataset as in the outlier run ===
    print(f"\nStep 1: Re-generating time-series data...")
    state_ts_raw, input_ts_raw, noise_w_raw = generate_time_series_data(
        system_params=TRUE_PARAMS, timesteps=T, 
        output_path=GENERATED_DATA_DIR, 
        base_filename=f"debug_run_T{T}_seed{data_seed}", 
        noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W}, 
        seed=data_seed
    )
    state_ts = np.array([state_ts_raw.flatten()])
    input_ts = np.array([input_ts_raw.flatten()])
    
    X_plus, X_minus, U_minus = state_ts[:, 1:T+1], state_ts[:, :T], input_ts[:, :T]

    # === 3. Analyze the generated data ===
    print("\n--- Analysis of the Input Data ---")
    Z_reg = np.vstack([X_minus, U_minus])
    try:
        # Check the condition number. A high number indicates potential numerical instability.
        condition_number_Z = np.linalg.cond(Z_reg)
        condition_number_ZZT = np.linalg.cond(Z_reg @ Z_reg.T)
        print(f"Condition number of Z_reg: {condition_number_Z:.4f}")
        print(f"Condition number of (Z*Z^T): {condition_number_ZZT:.4f}")
        if condition_number_ZZT > 1000:
            print(">>> WARNING: Matrix is potentially ill-conditioned <<<")
    except np.linalg.LinAlgError:
        print("Matrix Z or Z*Z^T is singular, which will cause issues.")
    
    # Check for correlation between input and noise (should be low)
    correlation = np.corrcoef(input_ts.flatten(), noise_w_raw.flatten())[0, 1]
    print(f"Correlation between input u and noise w: {correlation:.4f}")

    # Check the energy (L2 norm) of the signals
    energy_u = np.linalg.norm(input_ts)
    energy_w = np.linalg.norm(noise_w_raw)
    print(f"Signal Energy (u): {energy_u:.4f}")
    print(f"Noise Energy (w):  {energy_w:.4f}")
    if energy_u < energy_w:
        print(">>> WARNING: Noise energy is higher than signal energy! <<<")
    print("-----------------------------------------")


    # === 4. Calculate the Phi matrix ===
    c_delta = chi2.ppf(1 - CONFIDENCE_DELTA, df=DEGREES_OF_FREEDOM)
    Phi11 = (NOISE_STD_DEV_W**2) * c_delta * np.eye(1); Phi12 = np.zeros((1, T)); Phi21 = Phi12.T
    try:
        Phi22 = -np.linalg.pinv(Z_reg) @ Z_reg
        Phi = np.block([[Phi11, Phi12], [Phi21, Phi22]])
    except np.linalg.LinAlgError:
        print(f"Error: Could not compute Phi matrix for T={T}. Aborting.")
        return

    # === 5. Calculate the QMI Ellipse ===
    qmi_results = calculate_ellipse_from_qmi(
        X_plus, X_minus, U_minus, Phi11, Phi12, Phi21, Phi22
    )
    if qmi_results is None:
        print("QMI Ellipse calculation failed. Aborting.")
        return
        
    # === 6. Analyze and Print Final Metrics ===
    ellipse = ConfidenceEllipse(center=qmi_results['center'], p_matrix=qmi_results['shape_matrix'])
    print("\n--- Final Metrics for this Run ---")
    print(f"  - Estimated Center (a_hat, b_hat): ({ellipse.center[0]:.5f}, {ellipse.center[1]:.5f})")
    print(f"  - Area: {ellipse.area():.6f}")
    print(f"  - WCD: {ellipse.worst_case_deviation():.6f}")
    devs = ellipse.axis_parallel_deviations()
    print(f"  - Max Dev a/b: ({devs['max_dev_a']:.5f}, {devs['max_dev_b']:.5f})")
    print("----------------------------------")

    # === 7. Generate the plot ===
    print("\nStep 2: Generating final plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "debug_runs")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, f"debug_run_T{T}_seed{data_seed}.png")

    plot_qmi_ellipse(
        true_params=tuple(TRUE_PARAMS.values()),
        ellipse_center=qmi_results['center'],
        shape_matrix=qmi_results['shape_matrix'],
        T=T,
        output_path=plot_path
    )

if __name__ == '__main__':
    debug_single_run(T=T_TO_DEBUG, data_seed=SEED_TO_DEBUG)