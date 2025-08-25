"""
Experiment script for a single run of the original Set Membership method
using an adaptive grid search and MVEE calculation.
"""

import os
import numpy as np
from tqdm import tqdm
from scipy.stats import chi2
from src.data_generation import generate_time_series_data
from src.set_membership import find_feasible_set, calculate_mvee
from src.plotting import plot_mvee_with_points

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated') 


def run_set_membership_experiment(): 
    """Orchestrates a single run of the Set Membership analysis."""
    print("--- Starting Single Run: Set Membership Experiment ---")

    # === 3. Central Configuration for this run ===
    T = 50 # Number of data points to use
    DATA_SEED = 2025
    TRUE_PARAMS = (0.5, 0.5)
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    CONFIDENCE_DELTA = 0.05
    DEGREES_OF_FREEDOM = 2

    # Parameters for the adaptive grid search
    INITIAL_GRID_DENSITY = 300
    TARGET_PAIR_COUNT = 3000
    MIN_PAIRS, MAX_PAIRS = 2000, 4000
    MAX_ATTEMPTS = 5

    # === 4. Generate data for this run ===
    print(f"\nStep 1: Generating time-series data with T={T}...")
    state_data, input_data, _ = generate_time_series_data(
        system_params={'a': TRUE_PARAMS[0], 'b': TRUE_PARAMS[1]}, timesteps=T,
        output_path=GENERATED_DATA_DIR, # KORRIGIERT: Gültigen Pfad verwenden
        base_filename=f"temp_set_membership_T{T}", # KORRIGIERT: Gültigen Namen verwenden
        noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W},
        seed=DATA_SEED
    )
    X_plus, X_minus, U_minus = state_data[:, 1:T+1], state_data[:, 0:T], input_data[:, 0:T]

    # === 5. Calculate the Phi matrix ===
    c_delta = chi2.ppf(1 - CONFIDENCE_DELTA, df=DEGREES_OF_FREEDOM)
    Phi11 = (NOISE_STD_DEV_W**2) * c_delta * np.eye(1)
    Phi12 = np.zeros((1, T)); Phi21 = Phi12.T
    Z_reg = np.vstack([X_minus, U_minus])
    try:
        Phi22 = -np.linalg.pinv(Z_reg) @ Z_reg
        Phi = np.block([[Phi11, Phi12], [Phi21, Phi22]])
    except np.linalg.LinAlgError:
        print(f"Error: Could not compute Phi matrix for T={T}. Aborting.")
        return

    # === 6. Perform Adaptive Grid Search
    current_a_range, current_b_range = (0.49, 0.51), (0.49, 0.51)
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
            print(f"  (Attempt {attempt+1}) Found {num_pairs} pairs. Success.")
            break
        elif num_pairs < MIN_PAIRS:
            print(f"  (Attempt {attempt+1}) Too few pairs ({num_pairs}). Increasing density...")
            current_grid_density = int(current_grid_density * 1.5)
        else: # num_pairs > MAX_PAIRS
            print(f"  (Attempt {attempt+1}) Too many pairs ({num_pairs}). Reducing density...")
            current_grid_density = int(current_grid_density * 0.8)
    else:
        print(f"Warning: Could not find optimal number of pairs. Using {num_pairs} pairs.")
        
    # === 7. Calculate the MVEE ===
    mvee_results = calculate_mvee(valid_pairs)
    if mvee_results is None:
        print("MVEE calculation failed. Aborting.")
        return

    # === 8. Generate the plot ===
    print("\nStep 2: Generating final plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "set_membership_single")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, f"set_membership_T{T}.png")

    plot_mvee_with_points(
        feasible_points=valid_pairs,
        mvee_results=mvee_results,
        true_params=TRUE_PARAMS,
        T=T,
        output_path=plot_path
    )

    print("\n--- Set Membership Single Run Finished Successfully! ---")