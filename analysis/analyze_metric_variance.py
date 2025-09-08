# reporting/analyze_metric_cariance.py

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# --- Add project root to Python path to allow imports from src ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.plotting import plot_multi_metric_comparison

# --- Configuration ---
RESULTS_DIR = os.path.join(project_root, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures', 'final_analysis')

# Define the paths to your two summary files
FILE_A050 = os.path.join(RESULTS_DIR, 'mc_final_comparison_summary_a050.csv')
FILE_A099 = os.path.join(RESULTS_DIR, 'mc_final_comparison_summary_a099.csv')

def create_variance_comparison_plot():
    """
    Loads summary data from two simulation runs and creates a comparison
    plot of the standard deviation of the axis-parallel deviations.
    """
    print("--- Creating Set Membership variance comparison plot ---")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- 1. Load the data ---
    try:
        df_a050 = pd.read_csv(FILE_A050, header=[0, 1], index_col=0)
        df_a099 = pd.read_csv(FILE_A099, header=[0, 1], index_col=0)
    except FileNotFoundError as e:
        print("Error: Could not find result file. Please ensure both files exist:")
        print(f"  - {FILE_A050}")
        print(f"  - {FILE_A099}")
        return

    # --- 2. Create the plot ---
    # We create a temporary DataFrame to make plotting easier
    plot_df = pd.DataFrame({
        'T': df_a050.index,
        'std_dev_a_050': df_a050[('set_membership_max_dev_a', 'std')],
        'std_dev_b_050': df_a050[('set_membership_max_dev_b', 'std')],
        'std_dev_a_099': df_a099[('set_membership_max_dev_a', 'std')],
        'std_dev_b_099': df_a099[('set_membership_max_dev_b', 'std')],
    })

    # Define the "recipe" for our flexible plotting function
    plot_configs = [
        {'col': 'std_dev_a_050', 'label': 'Std. Dev. of dev_a (a=0.5)', 'color': 'green', 'linestyle': '-'},
        {'col': 'std_dev_b_050', 'label': 'Std. Dev. of dev_b (a=0.5)', 'color': 'limegreen', 'linestyle': '--'},
        {'col': 'std_dev_a_099', 'label': 'Std. Dev. of dev_a (a=0.99)', 'color': 'purple', 'linestyle': '-'},
        {'col': 'std_dev_b_099', 'label': 'Std. Dev. of dev_b (a=0.99)', 'color': 'mediumorchid', 'linestyle': '--'}
    ]

    output_path = os.path.join(FIGURES_DIR, "set_membership_variance_comparison.png")
    
    plot_multi_metric_comparison(
    dataframe=plot_df,
    metric_configs=plot_configs,
    x_col='T',
    y_label='Standard Deviation of Max Deviation', 
    title='Variance Comparison vs. T',
    output_path=output_path,
    use_log_scale=True 
)
    print(f"-> Variance comparison plot saved to: {output_path}")

if __name__ == '__main__':
    create_variance_comparison_plot()