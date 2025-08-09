import os
import numpy as np


from src.data_generation import generate_time_series_data
from src.system_identification import estimate_least_squares_timeseries, perform_bootstrap_analysis
from src.plotting import plot_bootstrap_rectangle


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_bootstrap_dean_experiment():
    """
    Orchestrates the full Bootstrap Dean experiment:
    1. Generates the required time-series data.
    2. Performs an initial estimation.
    3. Runs the bootstrap analysis.
    4. Saves results and visualizes them.
    """
    print("--- Starting Bootstrap Dean Experiment ---")

    # ------Configuration
    T_DATA_POINTS_TO_USE = 400
    TRUE_PARAMS = (0.5, 0.5)  # (a_true, b_true)
    
    # data config
    DATA_SEED = 42
    SIGMA_W = 0.0115 
    SIGMA_U = 1.0
    
    # Bootstrap config
    BOOTSTRAP_ITERATIONS = 2000
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_SEED = 123
    
    # === 4. Daten generieren ===
    print(f"\nStep 1: Generating time-series data with T={T_DATA_POINTS_TO_USE}...")
    generation_config = {
        'system_params': {'a': TRUE_PARAMS[0], 'b': TRUE_PARAMS[1]},
        'timesteps': T_DATA_POINTS_TO_USE,
        'output_path': GENERATED_DATA_DIR,
        'base_filename': f'timeseries_for_bootstrap_T{T_DATA_POINTS_TO_USE}',
        'noise_config': {'distribution': 'gaussian', 'std_dev': SIGMA_W},
        'seed': DATA_SEED
    }
    state_data_raw, input_data_raw, _ = generate_time_series_data(**generation_config)
    # Unsere LS-Funktion erwartet die Form (N, T), also fügen wir eine Dimension für N=1 hinzu
    state_data = np.array([state_data_raw.flatten()])
    input_data = np.array([input_data_raw.flatten()])

    # === 5. Initiale Schätzung durchführen ===
    print("\nStep 2: Performing initial LS estimation...")
    A_hat, B_hat = estimate_least_squares_timeseries(state_data, input_data)
    initial_estimate = (A_hat, B_hat)
    print(f"-> Initial estimate: A_hat = {A_hat.item():.6f}, B_hat = {B_hat.item():.6f}")
    
    # === 6. Bootstrap-Analyse durchführen ===
    bootstrap_results = perform_bootstrap_analysis(
        initial_estimate=initial_estimate,
        data_shape=(state_data.shape[0], T_DATA_POINTS_TO_USE), # (N, T)
        sigmas={'u': SIGMA_U, 'w': SIGMA_W},
        M=BOOTSTRAP_ITERATIONS,
        delta=CONFIDENCE_DELTA,
        seed=BOOTSTRAP_SEED
    )
    print(f"-> Bootstrap analysis complete.")
    print(f"-> Epsilon A: {bootstrap_results['epsilon_A']:.6f}, Epsilon B: {bootstrap_results['epsilon_B']:.6f}")

    # === 7. Ergebnisse speichern ===
    print("\nStep 3: Saving results...")
    output_filename = f"bootstrap_dean_T{T_DATA_POINTS_TO_USE}_M{BOOTSTRAP_ITERATIONS}.npz"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    np.savez(
        output_path,
        a_hat=A_hat.item(),
        b_hat=B_hat.item(),
        epsilon_A=bootstrap_results['epsilon_A'],
        epsilon_B=bootstrap_results['epsilon_B'],
        T=T_DATA_POINTS_TO_USE,
        M=BOOTSTRAP_ITERATIONS,
        delta=CONFIDENCE_DELTA
    )
    print(f"-> Results saved to {output_path}")

    # === 8. Visualisierung ===
    print("\nStep 4: Generating plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "bootstrap_dean_confidence_region.png")

    plot_bootstrap_rectangle(
        true_params=TRUE_PARAMS,
        estimated_params=(A_hat.item(), B_hat.item()),
        epsilons=(bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B']),
        confidence_delta=CONFIDENCE_DELTA,
        output_path=plot_path
    )

    print("\n--- Bootstrap Dean Experiment Finished ---")

