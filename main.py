# main.py
import argparse

# --- Import all available experiment functions ---
from experiments.run_data_dependent_bounds import run_data_dependent_bounds_experiment
from experiments.run_bootstrap_dean import run_bootstrap_dean_experiment
from experiments.run_set_membership import run_set_membership_experiment_over_T
from experiments.run_final_comparison import run_final_comparison_experiment
from experiments.run_sysid_comparison import run_sysid_comparison_experiment
from experiments.run_mc_sysid_comparison import run_monte_carlo_sysid_comparison # <-- Der wichtige Import

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
        choices=['dd-bounds', 'bootstrap-dean', 'set-membership', 
                 'final-comparison', 'sysid-compare', 'mc-sysid-compare'], # <-- Die wichtige Ergänzung in der Liste
        help="The name of the experiment to run."
    )
    
    # Define optional arguments for specific experiments
    parser.add_argument(
        "--runs", 
        type=int, 
        default=10, 
        help="Number of Monte Carlo runs to perform (used by 'mc-sysid-compare')."
    )

    args = parser.parse_args()

    # Dispatcher: Call the correct function based on the user's choice
    print(f"--- Starting experiment: '{args.experiment}' ---")
    
    if args.experiment == 'dd-bounds':
        run_data_dependent_bounds_experiment()
    elif args.experiment == 'bootstrap-dean':
        run_bootstrap_dean_experiment()
    elif args.experiment == 'set-membership':
        run_set_membership_experiment_over_T()
    elif args.experiment == 'final-comparison':
        run_final_comparison_experiment()
    elif args.experiment == 'sysid-compare':
        run_sysid_comparison_experiment(data_seed=0)
    elif args.experiment == 'mc-sysid-compare': # <-- Der wichtige neue Block
        run_monte_carlo_sysid_comparison(num_mc_runs=args.runs)
    else:
        # This case should not be reachable due to the 'choices' constraint
        print(f"Error: Unknown experiment '{args.experiment}'")

if __name__ == "__main__":
    main()