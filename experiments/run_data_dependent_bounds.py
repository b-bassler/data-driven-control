import os
import numpy as np

from src.data_generation import generate_iid_samples
from src.system_identification import estimate_least_squares_iid
from src.analysis import calculate_p_matrix_for_confidence_ellipse, analyze_ellipse_geometry
from src.plotting import plot_confidence_ellipse_from_matrix
# Pfad-Definitionen (angepasst für die neue Position)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')



def run_data_dependent_bounds_experiment():
    """
    A self-contained experiment that first generates I.I.D. data
    and then runs the data-dependent bounds analysis on it.
    """
    print("--- Starting Full Data-Dependent Bounds Experiment ---")

    # === 1. Zentrale Konfiguration für das gesamte Experiment ===
    # Parameter für die Datengenerierung
    N_SAMPLES = 10000
    DATA_SEED = 42
    BASE_FILENAME = 'iid_run_for_dd_bounds'
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    NOISE_STD_DEV = np.sqrt((0.01**2) / 3)
    
    # Parameter für die Analyse
    T_DATA_POINTS = 100  # Number of data points to use from the generated set
    CONFIDENCE_DELTA = 0.05  # 95% confidence

    # === 2. Daten generieren (aus data_generation) ===
    print(f"\nStep 1: Generating {N_SAMPLES} I.I.D. data samples...")
    generation_config = {
        'N': N_SAMPLES,
        'system_params': TRUE_PARAMS,
        'params_config': {
            'x_std_dev': 1.0,
            'u_std_dev': 1.0,
            'w_std_dev': NOISE_STD_DEV
        },
        'output_path': GENERATED_DATA_DIR,
        'base_filename': BASE_FILENAME,
        'seed': DATA_SEED
    }
    # Wir rufen die Funktion auf und fangen ihre Rückgabewerte direkt auf.
    # Die Funktion speichert die Daten trotzdem im Hintergrund für eine spätere Analyse.
    x_samples, u_samples, w_samples, y_samples = generate_iid_samples(**generation_config)
    
    # === 3. Daten für die Analyse vorbereiten ===
    print(f"\nStep 2: Preparing first {T_DATA_POINTS} data points for analysis...")
    x, u, y = x_samples[:T_DATA_POINTS], u_samples[:T_DATA_POINTS], y_samples[:T_DATA_POINTS]

    # === 4. Schätzung durchführen (aus system_identification) ===
    print("\nStep 3: Performing least-squares estimation...")
    A_est_mat, B_est_mat = estimate_least_squares_iid(x, u, y)
    if A_est_mat is None:
        print("Estimation failed. Aborting experiment.")
        return
    estimated_params = (A_est_mat[0, 0], B_est_mat[0, 0])
    print(f"-> Estimated Parameters: a_hat = {estimated_params[0]:.4f}, b_hat = {estimated_params[1]:.4f}")

    # === 5. Analyse durchführen (aus analysis) ===
    print("\nStep 4: Analyzing confidence ellipse...")
    # Zuerst die P-Matrix berechnen
    p_matrix = calculate_p_matrix_for_confidence_ellipse(x, u, NOISE_STD_DEV, CONFIDENCE_DELTA)
    # Dann die Geometrie daraus ableiten
    ellipse_properties = analyze_ellipse_geometry(p_matrix)
    print(f"-> Analysis complete. Worst-case deviation: {ellipse_properties['worst_case_deviation']:.4f}")

    # === 6. Ergebnisse speichern ===
    print("\nStep 5: Saving analysis results...")
    results_filepath = os.path.join(RESULTS_DIR, "data_dependent_bounds_results.npz")
    np.savez(
        results_filepath,
        estimated_center=estimated_params,
        true_center=tuple(TRUE_PARAMS.values()),
        **ellipse_properties
    )
    print(f"-> Analysis results saved to: {results_filepath}")
    
    # === 7. Visualisierung (aus plotting) ===
    print("\nStep 6: Generating plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_filepath = os.path.join(figures_dir, "data_dependent_ellipse_dean.png")
    plot_confidence_ellipse_from_matrix(
        true_params=tuple(TRUE_PARAMS.values()),
        estimated_params=estimated_params,
        p_matrix=p_matrix,
        confidence_delta=CONFIDENCE_DELTA,
        T=T_DATA_POINTS,
        output_path=plot_filepath
    )

    print("\n--- Experiment finished successfully! ---")
