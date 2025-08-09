import argparse

from experiments.run_data_dependent_bounds import run_data_dependent_bounds_experiment
from experiments.run_bootstrap_dean import run_bootstrap_dean_experiment




def main():
    """
    Main entry point for all experiments.
    Parses command-line arguments and calls the corresponding experiment function.
    """
    parser = argparse.ArgumentParser(
        description="Run simulation and analysis experiments for the bachelor thesis."
    )
    
    
    parser.add_argument(
        "experiment", 
        choices=['dd-bounds', 'bootstrap-dean'], # Unsere zwei verfügbaren Experimente
        help="The name of the experiment to run."
    )
    

    args = parser.parse_args()

    # Die Weiche: Rufe die passende Funktion basierend auf dem Argument auf
    print(f"--- Starting experiment: '{args.experiment}' ---")
    if args.experiment == 'dd-bounds':
        run_data_dependent_bounds_experiment()
    elif args.experiment == 'bootstrap-dean':
        run_bootstrap_dean_experiment()
    else:
        # Dieser Fall sollte durch 'choices' eigentlich nie eintreten
        print(f"Error: Unknown experiment '{args.experiment}'")

if __name__ == "__main__":
    main()