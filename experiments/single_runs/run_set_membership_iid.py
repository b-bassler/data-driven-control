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
# This setup ensures that the script can locate the necessary source modules
# from within the project structure.
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from src.data_generation import generate_iid_samples
    from src.set_membership import calculate_ellipse_from_qmi
    from src.analysis import ConfidenceEllipse
    from src.plotting import plot_qmi_ellipse
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the project's 'src' directory is accessible.")
    sys.exit(1)


# === 2. Path and Directory Definitions ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'qmi_stochastic')
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GENERATED_DATA_DIR, exist_ok=True)


def run_qmi_iid_stochastic_experiment():
    """
    Orchestrates a single run of the QMI-based Set-Membership analysis.

    This function executes the following steps:
    1.  Configures system parameters and the statistical model.
    2.  Generates an i.i.d. dataset based on the configuration.
    3.  Constructs and solves the QMI to find the feasible parameter set (ellipse).
    4.  Analyzes and prints key metrics of the resulting confidence ellipse.
    5.  Saves the metrics and a visualization of the result.
    """
    print("--- Starting Single Run: QMI Ellipse (Stochastic Approach) ---")

    # === 3. Central Configuration for the Experiment ===
    N = 50
    DATA_SEED = 2025

    # True system parameters [a, b] for y = ax + bu + w
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}

    # Configuration for generating random variables
    PARAMS_CONFIG = {
        'x_std_dev': 1.0,
        'u_std_dev': 1.0,
        'w_std_dev': 0.1
    }

    # Statistical confidence level (e.g., 0.05 for 95% confidence)
    CONFIDENCE_DELTA = 0.05
    # Degrees of freedom for the chi-squared distribution: n*(n+m)
    DEGREES_OF_FREEDOM = 2

    # === 4. Data Generation ===
    print(f"\nStep 1: Generating {N} i.i.d. samples...")

    # The function returns data as column vectors (Nx1)
    x_col, u_col, _, y_col = generate_iid_samples(
        N=N,
        system_params=TRUE_PARAMS,
        params_config=PARAMS_CONFIG,
        output_path=GENERATED_DATA_DIR,
        base_filename=f"temp_qmi_stochastic_data_N{N}",
        seed=DATA_SEED
    )

    # Reshape data into row vectors (1xN) as required by the QMI formulation
    X_plus = y_col.T
    X_minus = x_col.T
    U_minus = u_col.T

    # === 5. QMI Ellipse Calculation ===
    print("\nStep 2: Calculating ellipse from QMI (stochastic formulation)...")

    # Construct the components of the Phi matrix per Waarde et al. (2023), Thm. 5.8
    c_delta = chi2.ppf(1 - CONFIDENCE_DELTA, df=DEGREES_OF_FREEDOM)
    
    # Phi11 encodes the size of the confidence region based on the chi-squared distribution
    Phi11 = (PARAMS_CONFIG['w_std_dev']**2) * c_delta * np.eye(X_minus.shape[0])
    Phi12 = np.zeros((X_minus.shape[0], N))
    Phi21 = Phi12.T
    
    # Construct the regressor matrix Z
    Z_reg = np.vstack([X_minus, U_minus])
    
    # Phi22 is constructed to shape the QMI solution set into the correct
    # statistical confidence ellipsoid for the system parameters (A,B).
    try:
        Phi22 = -np.linalg.pinv(Z_reg) @ Z_reg
    except np.linalg.LinAlgError:
        print(f"Error: Could not compute the projection matrix for Phi22. Aborting.")
        return

    # Compute the ellipse parameters from the QMI components
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
        center=qmi_results['center'],
        p_matrix=qmi_results['shape_matrix']
    )
    metrics = {
        'N': N,
        'center_a': ellipse.center[0],
        'center_b': ellipse.center[1],
        'area': ellipse.area(),
        'wcd': ellipse.worst_case_deviation(),
        'max_dev_a': ellipse.axis_parallel_deviations()['max_dev_a'],
        'max_dev_b': ellipse.axis_parallel_deviations()['max_dev_b']
    }

    for key, value in metrics.items():
        print(f"   - {key}: {value:.5f}")

    # === 7. Save Results and Plot ===
    print("\nStep 4: Saving results and generating plot...")
    results_path = os.path.join(RESULTS_DIR, f"qmi_ellipse_iid_N{N}.npz")
    np.savez(results_path, **metrics)
    print(f"-> Metrics saved to: {results_path}")

    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, f"qmi_ellipse_iid_N{N}.png")

    plot_qmi_ellipse(
        true_params=tuple(TRUE_PARAMS.values()),
        ellipse_center=qmi_results['center'],
        shape_matrix=qmi_results['shape_matrix'],
        T=N, # Using T for plot label consistency
        output_path=plot_path
    )

    print("\n--- QMI Ellipse Experiment (Stochastic) Finished Successfully! ---")


if __name__ == '__main__':
    run_qmi_iid_stochastic_experiment()