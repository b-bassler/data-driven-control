"""
Experiment script for the Set Membership method, analyzing its performance
over a range of data points (T). Implements an adaptive grid search
to manage computational complexity.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import chi2

# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_time_series_data
from src.set_membership import find_feasible_set, calculate_mvee
from src.analysis import MVEEllipse
from src.plotting import plot_metric_trend

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_set_membership_experiment_over_T():
    """Orchestrates the Set Membership analysis over a range of T."""
    print("--- Starting Set Membership Experiment over T ---")

    # === 3. Central Configuration ===
    T_MAX = 100
    T_RANGE = np.arange(40, T_MAX + 1, 40)  # e.g., [40, 80, ..., 500]
    
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    DATA_SEED = 2025

    # Parameters for the Set Membership method
    CONFIDENCE_DELTA = 0.05
    DEGREES_OF_FREEDOM = 2  # k

    # Parameters for the adaptive grid search
    INITIAL_GRID_DENSITY = 100
    MIN_PAIRS = 2000  # Target minimum number of points for the optimizer
    MAX_PAIRS = 5000  # Target maximum to keep runtime reasonable
    MAX_ATTEMPTS = 20  # Max attempts to find a suitable number of points

    # === 4. Generate a single, large dataset ===
    print(f"\nStep 1: Generating one large time-series dataset with T={T_MAX}...")
    state_data_full, input_data_full, _ = generate_time_series_data(
        system_params=TRUE_PARAMS, timesteps=T_MAX,
        output_path=GENERATED_DATA_DIR, base_filename='set_membership_dataset',
        noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W},
        seed=DATA_SEED
    )

    # === 5. Loop over T and perform adaptive analysis ===
    results_list = []
    print(f"\nStep 2: Running analysis for T in {T_RANGE}...")
    for T in tqdm(T_RANGE, desc="Set Membership Progress"):
        # Prepare the data slice for the current iteration
        X_plus = state_data_full[:, 1:T+1]
        X_minus = state_data_full[:, 0:T]
        U_minus = input_data_full[:, 0:T]

        # Pre-calculate the Phi matrix based on the current data slice
        c_delta = chi2.ppf(1 - CONFIDENCE_DELTA, df=DEGREES_OF_FREEDOM)
        
        # Correctly construct the (T+1, T+1) Phi matrix as in the original script
        Phi11 = (NOISE_STD_DEV_W**2) * c_delta * np.eye(X_minus.shape[0])
        Phi12 = np.zeros((X_minus.shape[0], X_minus.shape[1]))
        Phi21 = Phi12.T
        
        Z_reg = np.vstack([X_minus, U_minus])
        try:
            # The correct formulation for Phi22 is -Z⁺Z
            Phi22 = -np.linalg.pinv(Z_reg) @ Z_reg
        except np.linalg.LinAlgError:
            print(f"Warning: Could not compute Phi22 for T={T}. Skipping.")
            continue

        Phi = np.block([[Phi11, Phi12], 
                        [Phi21, Phi22]])
        
        # --- Adaptive Grid Search Logic ---
        current_a_range, current_b_range = (0.48, 0.52), (0.48, 0.52)
        current_grid_density = INITIAL_GRID_DENSITY
        valid_pairs = np.array([])

        for attempt in range(MAX_ATTEMPTS):
            valid_pairs = find_feasible_set(
                X_plus, X_minus, U_minus, Phi,
                a_range=current_a_range, b_range=current_b_range,
                grid_density=current_grid_density
            )
            num_pairs = len(valid_pairs)
            if MIN_PAIRS <= num_pairs <= MAX_PAIRS:
                break
            elif num_pairs < MIN_PAIRS:
                current_grid_density = int(current_grid_density * 1.5)
            else: # num_pairs > MAX_PAIRS
                current_grid_density = int(current_grid_density * 0.8)
        else:
            print(f"Warning: Could not find optimal number of pairs for T={T}. Using {num_pairs} pairs.")

        # --- MVEE Calculation and Metric Analysis ---
        mvee_results = calculate_mvee(valid_pairs)
        if mvee_results is None:
            continue

        mvee_ellipse = MVEEllipse(mvee_results)
        
        results_list.append({
            'T': T,
            'set_membership_area': mvee_ellipse.area(),
            'set_membership_wcd': mvee_ellipse.worst_case_deviation(),
            'set_membership_max_dev_a': mvee_ellipse.axis_parallel_deviations()['max_dev_a'],
        })

    # === 6. Process and save the final results ===
    if not results_list:
        print("\nNo results were collected. Cannot process or plot.")
        return
        
    print("\nStep 3: Processing and saving collected data...")
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(RESULTS_DIR, "set_membership_over_T.csv")
    results_df.to_csv(results_path, index=False)
    print(f"-> Full results for Set Membership saved to {results_path}")
    
    # === 7. Generate trend plots for the collected metrics ===
    print("\nStep 4: Generating trend plots...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "set_membership")
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_metric_trend(results_df, 'set_membership_area', 'Area of Feasible Set', 'Set Membership: Area vs. T', os.path.join(figures_dir, "trend_area.png"))
    plot_metric_trend(results_df, 'set_membership_wcd', 'Worst-Case Deviation', 'Set Membership: WCD vs. T', os.path.join(figures_dir, "trend_wcd.png"))
    plot_metric_trend(results_df, 'set_membership_max_dev_a', 'Max Deviation for Parameter a', 'Set Membership: Max Dev for "a" vs. T', os.path.join(figures_dir, "trend_dev_a.png"))

    print("\n--- Set Membership Experiment Finished Successfully! ---")

