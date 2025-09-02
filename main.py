# main.py
import argparse
import os

# --- Path Definitions ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- Import all available experiment functions ---
from experiments.run_data_dependent_bounds import run_data_dependent_bounds_experiment
from experiments.run_bootstrap_dean import run_bootstrap_dean_experiment
from experiments.run_set_membership import run_set_membership_experiment
from experiments.run_qmi_analysis import run_qmi_analysis_experiment
from experiments.run_final_comparison import run_final_comparison_experiment
from experiments.run_mc_final_comparison import run_monte_carlo_final_comparison
from experiments.run_coverage_validation import perform_coverage_run
from experiments.run_coverage_validation_over_T import run_coverage_validation_over_T
from experiments.run_dd_bounds_calibration import run_dd_bounds_calibration_experiment
from experiments.run_bootstrap_validation import run_bootstrap_validation_experiment
from experiments.run_bootstrap_iid_validation import run_bootstrap_iid_validation_experiment
from experiments.run_mc_sysid_comparison import run_monte_carlo_sysid_methods
from experiments.run_coverage_variance_analysis import run_coverage_variance_analysis


def main():
    """
    Main entry point for all experiments.
    Parses command-line arguments and calls the corresponding experiment function.
    """
    parser = argparse.ArgumentParser(
        description="Run simulation and analysis experiments for the bachelor thesis."
    )
    
    # Define the main argument that selects the experiment
    parser.add_argument(
        "experiment", 
        choices=['dd-bounds', 'bootstrap-dean', 'set-membership', 'qmi-ellipse',
                 'sysid-compare', 'final-comparison', 
                 'mc-sysid-compare', 'mc-final-compare', 'coverage-test',
                 'coverage-over-t', 'calibrate-dd', 'bootstrap-validation',
                 'bootstrap-iid-validation', 'sysid-methods-compare',
                 'coverage-variance'], 
        help="The name of the experiment to run."
    )
    
    # Define optional arguments for specific experiments
    parser.add_argument(
        "--runs", 
        type=int, 
        default=10, 
        help="Number of Monte Carlo runs to perform."
    )

    args = parser.parse_args()

    # --- Directory Setup ---
    print("--- Ensuring all output directories exist... ---")
    required_dirs = [
        os.path.join(DATA_DIR, "generated"),
        os.path.join(RESULTS_DIR, "figures", "final_comparison"),
        os.path.join(RESULTS_DIR, "figures", "mc_sysid_comparison"),
        os.path.join(RESULTS_DIR, "figures", "mc_final_comparison"),
        os.path.join(RESULTS_DIR, "figures", "set_membership"),
        os.path.join(RESULTS_DIR, "figures", "sysid_comparison"),
        os.path.join(RESULTS_DIR, "figures", "qmi_ellipse"),
        os.path.join(RESULTS_DIR, "figures", "coverage_validation"),
        os.path.join(RESULTS_DIR, "figures", "calibration"),
    ]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # --- Dispatcher: Call the correct function based on the user's choice ---
    print(f"\n--- Starting experiment: '{args.experiment}' ---")
    
    if args.experiment == 'dd-bounds':
        run_data_dependent_bounds_experiment()
    elif args.experiment == 'bootstrap-dean':
        run_bootstrap_dean_experiment()
    elif args.experiment == 'set-membership':
        run_set_membership_experiment()
    elif args.experiment == 'qmi-ellipse':
        run_qmi_analysis_experiment()
    elif args.experiment == 'sysid-compare':
        run_final_comparison_experiment(data_seed=0)
    elif args.experiment == 'mc-final-compare':
        run_monte_carlo_final_comparison(num_mc_runs=args.runs)
    elif args.experiment == 'coverage-test': 
        perform_coverage_run(T=100, num_mc_runs=args.runs)
    elif args.experiment == 'coverage-over-t':
        run_coverage_validation_over_T()
    elif args.experiment == 'calibrate-dd':
        run_dd_bounds_calibration_experiment()
    elif args.experiment == 'bootstrap-validation':
        run_bootstrap_validation_experiment()
    elif args.experiment == 'bootstrap-iid-validation':
        run_bootstrap_iid_validation_experiment()
    elif args.experiment == 'sysid-methods-compare':
        run_monte_carlo_sysid_methods(num_mc_runs=args.runs)
    elif args.experiment == 'coverage-variance': 
        run_coverage_variance_analysis()
    else:
        print(f"Error: Unknown experiment '{args.experiment}'")

if __name__ == "__main__":
    main()