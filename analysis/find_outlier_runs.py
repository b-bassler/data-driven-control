# reporting/find_outlier_runs.py

import os
import pandas as pd
import sys

# --- Configuration ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(project_root, 'results')
RAW_DATA_FILE = os.path.join(RESULTS_DIR, 'mc_final_comparison_raw.csv')

# --- Parameters for the Investigation ---
# The T-value where you observed high variance in your plots
T_TO_INVESTIGATE = 70 
# The column containing the point estimate we want to check
METRIC_OF_INTEREST = 'set_membership_b_hat' 

def find_outlier_runs():
    """
    Loads the raw Monte Carlo results and identifies the runs with the
    minimum and maximum point estimates for a given metric and T-value.
    """
    print(f"--- Searching for outlier runs in: {RAW_DATA_FILE} ---")

    try:
        df = pd.read_csv(RAW_DATA_FILE)
    except FileNotFoundError:
        print(f"\n[ERROR] Raw data file not found at '{RAW_DATA_FILE}'")
        print("Please run the 'mc-final-compare' experiment first to generate the results.")
        return

    # Filter the DataFrame for the specific T value
    df_t = df[df['T'] == T_TO_INVESTIGATE].copy()
    
    if df_t.empty:
        print(f"\n[ERROR] No data found for T = {T_TO_INVESTIGATE}.")
        return
        
    if METRIC_OF_INTEREST not in df_t.columns:
        print(f"\n[ERROR] Metric '{METRIC_OF_INTEREST}' not found in the data file.")
        print(f"Available columns are: {list(df_t.columns)}")
        return

    # Find the index of the rows with the minimum and maximum values for our metric
    min_idx = df_t[METRIC_OF_INTEREST].idxmin()
    max_idx = df_t[METRIC_OF_INTEREST].idxmax()

    # Get the full data for these specific runs
    min_run = df_t.loc[min_idx]
    max_run = df_t.loc[max_idx]

    # Extract the seed (run_id) for both cases
    seed_for_min_b = int(min_run['run_id'])
    seed_for_max_b = int(max_run['run_id'])
    
    print(f"\n--- Investigation Results for T = {T_TO_INVESTIGATE} ---")
    
    print("\n[Run with SMALLEST b_hat]")
    print(min_run)
    print(f"\n>>> To reproduce this run, use data_seed = {seed_for_min_b} <<<")
    
    print("\n--------------------------------------------------")

    print("\n[Run with LARGEST b_hat]")
    print(max_run)
    print(f"\n>>> To reproduce this run, use data_seed = {seed_for_max_b} <<<")

if __name__ == '__main__':
    find_outlier_runs()