"""
Experiment script for a single run of the Tsiams data-dependent bounds method.
"""

import os
import numpy as np
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import all required tools
from src.data_generation import generate_time_series_data
from src.system_identification import estimate_least_squares_timeseries
from src.analysis import calculate_tsiams_ellipse_matrix, ConfidenceEllipse
from src.plotting import plot_tsiams_ellipse

# Define project paths
RESULTS_DIR = os.path.join(project_root, 'results')
GENERATED_DATA_DIR = os.path.join(project_root, 'data', 'generated')


def run_tsiams_analysis_experiment():
    """Orchestrates a single run of the Tsiams analysis."""
    print("--- Starting Single Run: Tsiams Data-Dependent Bounds ---")

    # === Configuration ===
    T = 100
    DATA_SEED = 2025
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    NOISE_STD_DEV_W = 0.1
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    
    # Parameters specific to the Tsiams method
    C = 0.00140625
    TAU = 2

    # === 1. Data Generation ===
    print(f"\nStep 1: Generating time-series data with T={T}...")
    state_data, input_data, _ = generate_time_series_data(
        system_params=TRUE_PARAMS, 
        timesteps=T, 
        output_path=GENERATED_DATA_DIR, # KORRIGIERT
        base_filename=f"temp_tsiams_data_T{T}", # KORRIGIERT
        noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W},
        seed=DATA_SEED
    )
    
    # === 2. Initial LS Estimation (to find the center) ===
    A_est, B_est = estimate_least_squares_timeseries(np.array([state_data.flatten()]), np.array([input_data.flatten()]))
    estimated_params = (A_est.item(), B_est.item())
    
    # === 3. Calculate Tsiams Ellipse Matrix ===
    print("\nStep 2: Calculating Tsiams ellipse matrix...")
    tsiams_results = calculate_tsiams_ellipse_matrix(
        state_data=state_data[:, :T], 
        input_data=input_data,
        true_A=np.array([[TRUE_PARAMS['a']]]), 
        true_B=np.array([[TRUE_PARAMS['b']]]),
        sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W},
        delta=CONFIDENCE_DELTA, 
        c=C, 
        tau=TAU
    )
    p_matrix = tsiams_results['p_matrix']
    
    # === 4. Generate Plot ===
    print("\nStep 3: Generating plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "tsiams_analysis")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, f"tsiams_ellipse_T{T}.png")

    plot_tsiams_ellipse(
        true_params=tuple(TRUE_PARAMS.values()),
        estimated_params=estimated_params,
        p_matrix=p_matrix, 
        T=T, 
        output_path=plot_path
    )
    
    print("\n--- Tsiams Experiment Finished Successfully! ---")