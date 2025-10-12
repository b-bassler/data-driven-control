"""
This script performs a single experimental run of the Set-Membership method
for a scalar linear system. It uses a direct Quadratic Matrix Inequality (QMI)
formulation derived from a stochastic noise model to compute a probabilistic
confidence set for the system parameters.
"""
import sys
import os
import numpy as np
from scipy.stats import chi2

# === 1. Project Setup and Module Imports ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_generation import generate_iid_samples
from src.set_membership import calculate_ellipse_from_qmi
from src.analysis import ConfidenceEllipse
from src.plotting import plot_qmi_ellipse


# === 2. Path and Directory Definitions ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# This directory is for method-specific results (like individual plots/metrics)
RESULTS_DIR_METHOD = os.path.join(BASE_DIR, 'results', 'qmi_stochastic')
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')

os.makedirs(RESULTS_DIR_METHOD, exist_ok=True)
os.makedirs(GENERATED_DATA_DIR, exist_ok=True)


def run_set_memnership_iid_experiment():
    """ Orchestrates a single run of the QMI-based Set-Membership analysis. """
    print("--- Starting Single Run: Set-Membership (I.I.D.) ---")

    # === 3. Central Configuration for the Experiment ===
    N = 200
    DATA_SEED = 2 

    TRUE_PARAMS = {'a': 0.99, 'b': 0.5}
    PARAMS_CONFIG = {
        'x_std_dev': 1.0, 'u_std_dev': 1.0, 'w_std_dev': 0.1
    }
    CONFIDENCE_DELTA = 0.05
    DEGREES_OF_FREEDOM = 2

    # === 4. Data Generation ===
    print(f"\nStep 1: Generating {N} i.i.d. samples...")
    x_col, u_col, _, y_col = generate_iid_samples(
        N=N, system_params=TRUE_PARAMS, params_config=PARAMS_CONFIG,
        output_path=GENERATED_DATA_DIR, base_filename=f"temp_qmi_stochastic_data_N{N}",
        seed=DATA_SEED
    )
    X_plus, X_minus, U_minus = y_col, x_col, u_col

    # === 5. QMI Ellipse Calculation ===
    print("\nStep 2: Calculating ellipse from QMI (stochastic formulation)...")
    c_delta = chi2.ppf(1 - CONFIDENCE_DELTA, df=DEGREES_OF_FREEDOM)
    Phi11 = (PARAMS_CONFIG['w_std_dev']**2) * c_delta * np.eye(X_minus.shape[0])
    Phi12 = np.zeros((X_minus.shape[0], N))
    Phi21 = Phi12.T
    Z_reg = np.vstack([X_minus, U_minus])
    try:
        Phi22 = -np.linalg.pinv(Z_reg) @ Z_reg
    except np.linalg.LinAlgError:
        print("Error: Could not compute the projection matrix for Phi22. Aborting.")
        return
    qmi_results = calculate_ellipse_from_qmi(
        X_plus, X_minus, U_minus, Phi11, Phi12, Phi21, Phi22
    )
    if qmi_results is None:
        print("Ellipse calculation failed. Aborting.")
        return
    print("-> Ellipse parameters calculated successfully.")

    # === 6. Analysis and Metrics ===
    print("\nStep 3: Analyzing ellipse metrics...")
    ellipse = ConfidenceEllipse(
        center=qmi_results['center'], p_matrix=qmi_results['shape_matrix']
    )
    
    # === 7. Save Individual Results and Plot ===
    print("\nStep 4: Saving individual results and generating plot...")
    
    # Generate the standard plot for this single run
    figures_dir = os.path.join(RESULTS_DIR_METHOD, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, f"qmi_ellipse_iid_N{N}.png")
    plot_qmi_ellipse(
        true_params=tuple(TRUE_PARAMS.values()),
        ellipse_center=qmi_results['center'],
        shape_matrix=qmi_results['shape_matrix'],
        T=N,
        output_path=plot_path
    )
    
    # === 8. Save Bound Geometry for Comparison Plot ===
    print("\nStep 5: Saving bound geometry for final comparison plot...")
    

    comparison_dir = os.path.join(BASE_DIR, "results", "comparison_data")
    os.makedirs(comparison_dir, exist_ok=True)
    bound_output_path = os.path.join(comparison_dir, f"bound_set_membership_iid_N{N}.npz")

    np.savez(
        bound_output_path,
        type=np.array('ellipse'),
        method=np.array('Set Membership (QMI)'),
        center=ellipse.center,
        p_matrix=ellipse.p_matrix
    )
    print(f"-> Bound geometry saved to: {bound_output_path}")

    print("\n--- QMI Ellipse Experiment Finished Successfully! ---")


if __name__ == '__main__':
    run_set_memnership_iid_experiment()
