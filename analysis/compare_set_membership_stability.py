# reporting/compare_set_membership_stability.py

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# --- Add project root to Python path to allow imports from src ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# --- Configuration ---
RESULTS_DIR = os.path.join(project_root, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures', 'final_analysis')

FILE_A050 = os.path.join(RESULTS_DIR, 'mc_final_comparison_summary_a050.csv')
FILE_A099 = os.path.join(RESULTS_DIR, 'mc_final_comparison_summary_a099.csv')

def create_stability_comparison_plot():
    """
    Loads results from two different simulation runs (stable vs. marginally stable)
    and creates a comparison plot for the Set Membership method's axis-parallel deviations.
    """
    print("--- Creating Set Membership stability comparison plot ---")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- 1. Load the data ---
    try:
        df_a050 = pd.read_csv(FILE_A050, header=[0, 1], index_col=0)
        df_a099 = pd.read_csv(FILE_A099, header=[0, 1], index_col=0)
    except FileNotFoundError as e:
        print("Error: Could not find one or both result files.")
        print("Please ensure you have run the 'mc-final-compare' experiment for both scenarios")
        print("and renamed the summary files correctly to:")
        print(f"  - {FILE_A050}")
        print(f"  - {FILE_A099}")
        print(f"Original error: {e}")
        return

    # --- 2. Create the plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # --- Plot data for a = 0.5 (stable system) ---
    ax.plot(df_a050.index, df_a050[('set_membership_max_dev_a', 'mean')], 
            label='Set Membership (a), a=0.5', color='green', linestyle='-', marker='s')
    ax.fill_between(df_a050.index, 
                    df_a050[('set_membership_max_dev_a', 'mean')] - df_a050[('set_membership_max_dev_a', 'std')], 
                    df_a050[('set_membership_max_dev_a', 'mean')] + df_a050[('set_membership_max_dev_a', 'std')],
                    color='green', alpha=0.15)
    
    ax.plot(df_a050.index, df_a050[('set_membership_max_dev_b', 'mean')], 
            label='Set Membership (b), a=0.5', color='limegreen', linestyle='--', marker='s')
    ax.fill_between(df_a050.index, 
                    df_a050[('set_membership_max_dev_b', 'mean')] - df_a050[('set_membership_max_dev_b', 'std')], 
                    df_a050[('set_membership_max_dev_b', 'mean')] + df_a050[('set_membership_max_dev_b', 'std')],
                    color='limegreen', alpha=0.15)

    # --- Plot data for a = 0.99 (marginally stable system) ---
    ax.plot(df_a099.index, df_a099[('set_membership_max_dev_a', 'mean')], 
            label='Set Membership (a), a=0.99', color='purple', linestyle='-', marker='o')
    ax.fill_between(df_a099.index, 
                    df_a099[('set_membership_max_dev_a', 'mean')] - df_a099[('set_membership_max_dev_a', 'std')], 
                    df_a099[('set_membership_max_dev_a', 'mean')] + df_a099[('set_membership_max_dev_a', 'std')],
                    color='purple', alpha=0.15)
    
    ax.plot(df_a099.index, df_a099[('set_membership_max_dev_b', 'mean')], 
            label='Set Membership (b), a=0.99', color='mediumorchid', linestyle='--', marker='o')
    ax.fill_between(df_a099.index, 
                    df_a099[('set_membership_max_dev_b', 'mean')] - df_a099[('set_membership_max_dev_b', 'std')], 
                    df_a099[('set_membership_max_dev_b', 'mean')] + df_a099[('set_membership_max_dev_b', 'std')],
                    color='mediumorchid', alpha=0.15)

    # --- Final plot styling ---
    ax.set_xlabel("Number of Data Points (T)")
    ax.set_ylabel("Mean Max Axis-Parallel Deviation (log scale)")
    ax.set_title("Effect of System Stability on Set Membership Bounds")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yscale('log')
    
    output_path = os.path.join(FIGURES_DIR, "set_membership_stability_comparison.png")
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"-> Comparison plot saved to: {output_path}")

if __name__ == '__main__':
    create_stability_comparison_plot()