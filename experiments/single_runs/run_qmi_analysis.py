"""
Experiment script for a single run of the Set Membership method using the
direct QMI derivation. This script calculates and plots one feasible set ellipse.
"""

import os
import numpy as np
from scipy.stats import chi2

# --- 1. Imports from our src library ---
from src.data_generation import generate_time_series_data
from src.set_membership import calculate_ellipse_from_qmi
from src.analysis import ConfidenceEllipse
from src.plotting import plot_qmi_ellipse 

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')

def run_qmi_analysis_experiment():
    """
    Orchestrates a single run of the QMI-based Set Membership analysis.
    """
    print("--- Starting Single Run: QMI-based Ellipse Analysis ---")

    # === 3. Central Configuration for this specific run 
    T = 50  # The number of data points to use for this run
    DATA_SEED = 2025
    
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    CONFIDENCE_DELTA = 0.05
    DEGREES_OF_FREEDOM = 2

    # === 4. Generate the dataset for this run
    print(f"\nStep 1: Generating time-series data with T={T}...")
    state_data, input_data, _ = generate_time_series_data(
        system_params=TRUE_PARAMS, timesteps=T,
        output_path=GENERATED_DATA_DIR,  
        base_filename=f"temp_qmi_data_T{T}",
        noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W},
        seed=DATA_SEED
    )
    
    # Prepare the data matrices as required by the formulas
    X_plus = state_data[:, 1:T+1]
    X_minus = state_data[:, 0:T]
    U_minus = input_data[:, 0:T]

    # === 5. Calculate the QMI Ellipse directly 
    print("\nStep 2: Calculating ellipse directly from QMI...")
    
    # Pre-calculate the Phi matrix components
    c_delta = chi2.ppf(1 - CONFIDENCE_DELTA, df=DEGREES_OF_FREEDOM)
    Phi11 = (NOISE_STD_DEV_W**2) * c_delta * np.eye(X_minus.shape[0])
    Phi12 = np.zeros((X_minus.shape[0], X_minus.shape[1]))
    Phi21 = Phi12.T
    Z_reg = np.vstack([X_minus, U_minus])
    try:
        Phi22 = -np.linalg.pinv(Z_reg) @ Z_reg
    except np.linalg.LinAlgError:
        print(f"Error: Could not compute Phi22 for T={T}. Aborting.")
        return
    
    # Call our dedicated tool function from src
    qmi_results = calculate_ellipse_from_qmi(
        X_plus, X_minus, U_minus, Phi11, Phi12, Phi21, Phi22
    )
    if qmi_results is None:
        print("Ellipse calculation failed. Aborting.")
        return
    
    print("-> Ellipse calculated successfully.")

    # === 6. Analyze Metrics using our ConfidenceEllipse class 
    ellipse = ConfidenceEllipse(
        center=qmi_results['center'], 
        p_matrix=qmi_results['shape_matrix']
    )
    metrics = {
        'T': T,
        'center_a': ellipse.center[0],
        'center_b': ellipse.center[1],
        'area': ellipse.area(),
        'wcd': ellipse.worst_case_deviation(),
        'max_dev_a': ellipse.axis_parallel_deviations()['max_dev_a'],
        'max_dev_b': ellipse.axis_parallel_deviations()['max_dev_b']
    }
    
    for key, value in metrics.items():
        print(f"  - {key}: {value:.5f}")

    # === 7. Save the results of this single run 
    print("\nStep 4: Saving results and plot...")
    results_path = os.path.join(RESULTS_DIR, f"qmi_ellipse_results_T{T}.npz")
    np.savez(results_path, **metrics)
    print(f"-> Metrics saved to: {results_path}")

    # === 8. Generate a plot of the resulting ellipse
    figures_dir = os.path.join(RESULTS_DIR, "figures", "qmi_ellipse")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, f"qmi_ellipse_T{T}.png")

    plot_qmi_ellipse(
        true_params=tuple(TRUE_PARAMS.values()),
        ellipse_center=qmi_results['center'],
        shape_matrix=qmi_results['shape_matrix'],
        T=T,
        output_path=plot_path
    )

    print("\n--- QMI Ellipse Experiment Finished Successfully! ---")