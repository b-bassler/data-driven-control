import os
from src.data_generation import generate_time_series_data

# --- 1. Define Project Paths (Portable and Robust) ---
# This part ensures that the script will work on any computer without
# needing to change the paths manually.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')


# --- 2. Define the "Comparison Run" Experiment ---
def run_uniform_generation_for_comparison():
    """
    Generates a dataset with parameters that are now identical 
    to the updated old uniform script for verification.
    """
    print("--- Starting Comparison Run for 'uniform' data generation ---")

    # This configuration dictionary exactly mimics the parameters of your updated old script.
    # We don't need to specify initial_state_config or input_config,
    # because the default behavior of our function now matches the script.
    comparison_config = {
        'system_params': {'a': 0.5, 'b': 0.5},
        'timesteps': 100000,
        'output_path': GENERATED_DATA_DIR,
        'base_filename': 'uniform_run_1', # A clear name for the output
        'noise_config': {'distribution': 'uniform', 'level': 0.01},
        'seed': 1
    }

    # Call our generation function with the defined configuration
    generate_time_series_data(**comparison_config)

    print("\n--- Comparison data generated successfully! ---")
    print("You can now run the verify.py script.")


# --- 3. Script Entry Point ---
if __name__ == "__main__":
    run_uniform_generation_for_comparison()