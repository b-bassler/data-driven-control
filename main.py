# main.py
import argparse
import os

# --- Path Definitions ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- Import all available experiment functions ---
from experiments.single_runs.run_dean_ddbounds_iid import run_data_dependent_bounds_experiment
from experiments.single_runs.run_bootstrap_trajectory import run_bootstrap_dean_trajectory
from experiments.single_runs.run_set_membership_trajectory import run_set_membership_iid_experiment
from experiments.single_runs.run_tsiams_ddbounds_trajectory import run_tsiams_analysis_experiment
from experiments.single_runs.run_set_membership_iid import run_set_memnership_iid_experiment
from experiments.single_runs.run_bootstrap_iid import run_bootstrap_dean_iid
from experiments.run_dd_bounds_calibration import run_dd_bounds_calibration_experiment
from experiments.run_coverage_iid import run_coverage_iid_over_T
from experiments.run_mc_trajectory import run_mc_trajectory_comparison
from experiments.run_mc_iid import run_mc_iid_comparison


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
        choices=['dd_bounds_trajectory', 'dd_bounds_iid','set_membership_iid', 
                 'set_membership_trajectory', 'bootstrap_iid','bootstrap_trajectory',  
                 'trajectory_comparison', 'iid_comparison', 'coverage_iid', 'dd_bounds_calibration'], 
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
    
    if args.experiment == 'dd_bounds_trajectory':
        run_data_dependent_bounds_experiment()
    elif args.experiment == 'dd_bounds_iid':
        run_tsiams_analysis_experiment()
    elif args.experiment == 'set_membership_iid':
        run_set_memnership_iid_experiment()
    elif args.experiment == 'set_membership_trajectory':
        run_set_membership_iid_experiment()
    elif args.experiment == 'bootstrap_iid': 
        run_bootstrap_dean_iid()
    elif args.experiment == 'bootstrap_trajectory':
        run_bootstrap_dean_trajectory()
    elif args.experiment == 'trajectory_comparison':
        run_mc_trajectory_comparison(num_mc_runs=args.runs)
    elif args.experiment == 'iid_comparison':
        run_mc_iid_comparison(num_mc_runs=args.runs)
    elif args.experiment == 'coverage_iid':
        run_coverage_iid_over_T()
    elif args.experiment == 'dd_bounds_calibration':
        run_dd_bounds_calibration_experiment()
    else:
        print(f"Error: Unknown experiment '{args.experiment}'")

if __name__ == "__main__":
    main()