# reporting/analyze_coefficient_of_variation.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Add project root to Python path to allow imports from src ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Assuming a flexible plotting function exists in your src library
from src.plotting import plot_multi_metric_comparison

# --- Configuration ---
RESULTS_DIR = os.path.join(project_root, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures', 'final_analysis')

# Define the paths to your two summary files from the Monte Carlo simulations
FILE_A050 = os.path.join(RESULTS_DIR, 'mc_final_comparison_summary_a050.csv')
FILE_A099 = os.path.join(RESULTS_DIR, 'mc_final_comparison_summary_a099.csv')

def analyze_and_plot_cv():
    """
    Loads Monte Carlo summary data, calculates the Coefficient of Variation (CV)
    for key uncertainty metrics, and generates comparison plots.

    The Coefficient of Variation (CV = std / mean) is a normalized measure of
    dispersion, which allows for a fair comparison of the variability of metrics
    across different scales.
    """
    print("--- Analyzing Coefficient of Variation (Relative Standard Deviation) ---")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- 1. Load the summary data ---
    try:
        df_a050 = pd.read_csv(FILE_A050, header=[0, 1], index_col=0)
        df_a099 = pd.read_csv(FILE_A099, header=[0, 1], index_col=0)
    except FileNotFoundError:
        print("Error: Could not find result files. Please ensure both files exist:")
        print(f"  - {FILE_A050}")
        print(f"  - {FILE_A099}")
        return

    # --- 2. Calculate the Coefficient of Variation (CV) for all methods and metrics ---
    metrics_to_analyze = [
        'set_membership_area', 'set_membership_wcd', 'set_membership_max_dev_a', 'set_membership_max_dev_b',
        'bootstrap_area', 'bootstrap_wcd', 'bootstrap_max_dev_a', 'bootstrap_max_dev_b',
        'dd_bounds_area', 'dd_bounds_wcd', 'dd_bounds_max_dev_a', 'dd_bounds_max_dev_b'
    ]
    
    cv_df_a050 = pd.DataFrame(index=df_a050.index)
    cv_df_a099 = pd.DataFrame(index=df_a099.index)

    for metric in metrics_to_analyze:
        mean_050 = df_a050.get((metric, 'mean'), pd.Series(np.nan, index=df_a050.index))
        std_050 = df_a050.get((metric, 'std'), pd.Series(np.nan, index=df_a050.index))
        cv_df_a050[metric] = std_050 / mean_050.replace(0, np.nan)

        mean_099 = df_a099.get((metric, 'mean'), pd.Series(np.nan, index=df_a099.index))
        std_099 = df_a099.get((metric, 'std'), pd.Series(np.nan, index=df_a099.index))
        cv_df_a099[metric] = std_099 / mean_099.replace(0, np.nan)

    # --- 3. Generate a plot for each key metric's CV ---
    
    # Plot 1: CV of the Area
    plot_configs_area = [
        {'df': 'a050', 'col': 'dd_bounds_area', 'label': 'Data-Dependent (a=0.50)', 'color': 'blue', 'linestyle': '-'},
        {'df': 'a050', 'col': 'bootstrap_area', 'label': 'Bootstrap (a=0.50)', 'color': 'orange', 'linestyle': '-'},
        {'df': 'a050', 'col': 'set_membership_area', 'label': 'Set Membership (a=0.50)', 'color': 'green', 'linestyle': '-'},
        {'df': 'a099', 'col': 'dd_bounds_area', 'label': 'Data-Dependent (a=0.99)', 'color': 'deepskyblue', 'linestyle': '--'},
        {'df': 'a099', 'col': 'bootstrap_area', 'label': 'Bootstrap (a=0.99)', 'color': 'gold', 'linestyle': '--'},
        {'df': 'a099', 'col': 'set_membership_area', 'label': 'Set Membership (a=0.99)', 'color': 'limegreen', 'linestyle': '--'}
    ]
    
    # Pass a dictionary of dataframes to the plotting function
    dataframes = {'a050': cv_df_a050, 'a099': cv_df_a099}
    
    output_path = os.path.join(FIGURES_DIR, "cv_comparison_area.png")
    plot_multi_metric_comparison(
        dataframes=dataframes, metric_configs=plot_configs_area, x_col='T',
        y_label='Coefficient of Variation (CV)',
        title='Relative Variability of Uncertainty Area vs. N',
        output_path=output_path, use_log_scale=False
    )
    print(f"-> CV comparison plot for Area saved to: {output_path}")

    # Plot 2: CV of the Worst-Case Deviation (WCD)
    plot_configs_wcd = [
        {'df': 'a050', 'col': 'dd_bounds_wcd', 'label': 'Data-Dependent (a=0.50)', 'color': 'blue', 'linestyle': '-'},
        {'df': 'a050', 'col': 'bootstrap_wcd', 'label': 'Bootstrap (a=0.50)', 'color': 'orange', 'linestyle': '-'},
        {'df': 'a050', 'col': 'set_membership_wcd', 'label': 'Set Membership (a=0.50)', 'color': 'green', 'linestyle': '-'},
        {'df': 'a099', 'col': 'dd_bounds_wcd', 'label': 'Data-Dependent (a=0.99)', 'color': 'deepskyblue', 'linestyle': '--'},
        {'df': 'a099', 'col': 'bootstrap_wcd', 'label': 'Bootstrap (a=0.99)', 'color': 'gold', 'linestyle': '--'},
        {'df': 'a099', 'col': 'set_membership_wcd', 'label': 'Set Membership (a=0.99)', 'color': 'limegreen', 'linestyle': '--'}
    ]
    output_path = os.path.join(FIGURES_DIR, "cv_comparison_wcd.png")
    plot_multi_metric_comparison(
        dataframes=dataframes, metric_configs=plot_configs_wcd, x_col='T',
        y_label='Coefficient of Variation (CV)',
        title='Relative Variability of Worst-Case Deviation vs. N',
        output_path=output_path, use_log_scale=False
    )
    print(f"-> CV comparison plot for WCD saved to: {output_path}")

# --- 3. Generate focused plots for Max Deviation CV ---

    # Plot 3a: Relative Variability of Parameter 'a' Estimate
    plot_configs_dev_a = [
        # Data-Dependent
        {'df': 'a050', 'col': 'dd_bounds_max_dev_a', 'label': 'Data-Dep. (a=0.50)', 'color': 'blue', 'linestyle': '-'},
        {'df': 'a099', 'col': 'dd_bounds_max_dev_a', 'label': 'Data-Dep. (a=0.99)', 'color': 'deepskyblue', 'linestyle': '--'},
        # Bootstrap
        {'df': 'a050', 'col': 'bootstrap_max_dev_a', 'label': 'Bootstrap (a=0.50)', 'color': 'orange', 'linestyle': '-'},
        {'df': 'a099', 'col': 'bootstrap_max_dev_a', 'label': 'Bootstrap (a=0.99)', 'color': 'gold', 'linestyle': '--'},
        # Set Membership
        {'df': 'a050', 'col': 'set_membership_max_dev_a', 'label': 'Set Memb. (a=0.50)', 'color': 'green', 'linestyle': '-'},
        {'df': 'a099', 'col': 'set_membership_max_dev_a', 'label': 'Set Memb. (a=0.99)', 'color': 'limegreen', 'linestyle': '--'},
    ]

    output_path_a = os.path.join(FIGURES_DIR, "cv_comparison_max_dev_a.png")
    plot_multi_metric_comparison(
        dataframes=dataframes, 
        metric_configs=plot_configs_dev_a, 
        x_col='T',
        y_label='Coefficient of Variation (CV)',
        title='Relative Variability of Parameter \'a\' Estimate',
        output_path=output_path_a, 
        use_log_scale=False,
    )
    print(f"-> CV comparison plot for Parameter 'a' saved to: {output_path_a}")

    # Plot 3b: Relative Variability of Parameter 'b' Estimate
    plot_configs_dev_b = [
    # Data-Dependent
    {'df': 'a050', 'col': 'dd_bounds_max_dev_b', 'label': 'Data-Dep. (a=0.50), dev_b', 'color': 'blue', 'linestyle': '-'},
    {'df': 'a099', 'col': 'dd_bounds_max_dev_b', 'label': 'Data-Dep. (a=0.99), dev_b', 'color': 'deepskyblue', 'linestyle': '--'},
    # Bootstrap
    {'df': 'a050', 'col': 'bootstrap_max_dev_b', 'label': 'Bootstrap (a=0.50), dev_b', 'color': 'orange', 'linestyle': '-'},
    {'df': 'a099', 'col': 'bootstrap_max_dev_b', 'label': 'Bootstrap (a=0.99), dev_b', 'color': 'gold', 'linestyle': '--'},
    # Set Membership
    {'df': 'a050', 'col': 'set_membership_max_dev_b', 'label': 'Set Memb. (a=0.50), dev_b', 'color': 'green', 'linestyle': '-'},
    {'df': 'a099', 'col': 'set_membership_max_dev_b', 'label': 'Set Memb. (a=0.99), dev_b', 'color': 'limegreen', 'linestyle': '--'},
    ]


    output_path_b = os.path.join(FIGURES_DIR, "cv_comparison_max_dev_b.png")
    plot_multi_metric_comparison(
        dataframes=dataframes, 
        metric_configs=plot_configs_dev_b, 
        x_col='T',
        y_label='Coefficient of Variation (CV)',
        title='Relative Variability of Parameter \'b\' Estimate',
        output_path=output_path_b, 
        use_log_scale=False,
    )
    print(f"-> CV comparison plot for Parameter 'b' saved to: {output_path_b}")

if __name__ == '__main__':
    analyze_and_plot_cv()

