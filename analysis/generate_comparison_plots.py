import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from typing import List, Dict, Any

# --- Add project root to Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Configuration ---
RESULTS_DIR = os.path.join(project_root, 'results')
COMPARISON_DATA_DIR = os.path.join(RESULTS_DIR, 'comparison_data')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures', 'final_comparison')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage[light]{firasans}
        \usepackage{amsmath}
    """,
})

# Define parameters consistent with the single runs to plot the true value
T = 100
TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
 
label_map = {
        'Data-Dependent (Dean)': 'Data-Dependent',
        'Data-Dependent (Tsiams)': 'Data-Dependent',
        'Bootstrap (I.I.D.)': 'Bootstrap',
        'Bootstrap (Trajectory)': 'Bootstrap',
        'Set Membership (QMI)': 'Set Membership',
        
    }
# Define consistent colors for the methods
COLOR_MAP = {
    'Data-Dependent (Dean)': 'blue',
    'Data-Dependent (Tsiams)': 'blue',
    'Bootstrap (I.I.D.)': 'red',
    'Bootstrap (Trajectory)': 'red',
    'Set Membership (QMI)': 'green',
}

def load_bound_data(file_list: List[str]) -> List[Dict[str, Any]]:
    """Loads a list of .npz files containing bound geometries."""
    loaded_bounds = []
    for filename in file_list:
        filepath = os.path.join(COMPARISON_DATA_DIR, filename)
        try:
            data = np.load(filepath)
            bound_info = {key: data[key].item() if data[key].ndim == 0 else data[key] for key in data}
            loaded_bounds.append(bound_info)
        except FileNotFoundError:
            print(f"Warning: Could not find data file '{filepath}'. Skipping.")
    return loaded_bounds

def plot_combined_bounds(bounds_list: List[Dict], true_params: tuple, title: str, output_path: str):
    """
    Plots multiple uncertainty bounds (ellipses, rectangles) on a single figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))


    label_map = {
        'Data-Dependent (Dean)': 'Data-Dependent',
        'Data-Dependent (Tsiams)': 'Data-Dependent',
        'Bootstrap (I.I.D.)': 'Bootstrap',
        'Bootstrap (Trajectory)': 'Bootstrap',
        'Set Membership (QMI)': 'Set Membership',

    }

    plotted_labels = set()

    for bound in bounds_list:
        method_name = bound['method']
        color = COLOR_MAP.get(method_name, 'gray') 
        center = bound['center']

        display_label = label_map.get(method_name, method_name)

        if display_label in plotted_labels:
            plot_label = None 
        else:
            plot_label = display_label
            plotted_labels.add(display_label)

        if bound['type'] == 'ellipse':
            p_matrix = bound['p_matrix']

            cov_matrix = np.linalg.inv(p_matrix)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            order = eigenvalues.argsort()[::-1]
            width = 2 * np.sqrt(eigenvalues[order[0]])
            height = 2 * np.sqrt(eigenvalues[order[1]])
            angle = np.degrees(np.arctan2(*eigenvectors[:, order[0]][::-1]))

            fill_patch = Ellipse(xy=center, width=width, height=height, angle=angle,
                                 facecolor=color, alpha=0.15, label=plot_label)
            ax.add_patch(fill_patch)
            border_patch = Ellipse(xy=center, width=width, height=height, angle=angle,
                                     edgecolor=color, facecolor='none', linewidth=2)
            ax.add_patch(border_patch)
           
        elif bound['type'] == 'rectangle':
            epsilons = bound['epsilons']
            
 
            fill_patch = Rectangle(xy=center - epsilons, width=2*epsilons[0], height=2*epsilons[1],
                                     facecolor=color, alpha=0.15, linestyle='--', label=plot_label)
            ax.add_patch(fill_patch)
            border_patch = Rectangle(xy=center - epsilons, width=2*epsilons[0], height=2*epsilons[1],
                                       edgecolor=color, facecolor='none', linewidth=2, linestyle='-')
            ax.add_patch(border_patch)
            
    # Plot true parameter on top of everything
    ax.plot(true_params[0], true_params[1], 'x', color='red', markersize=10, markeredgewidth=2.5, label='True Parameters')
    ax.plot(center[0], center[1], marker='+', color='black', markersize=12, markeredgewidth=2, label = "Least-squares")

    

    ax.set_title(title, fontsize=16) 
    ax.set_xlabel('Parameter a', fontsize=20) 
    ax.set_ylabel('Parameter b', fontsize=20) 


    ax.tick_params(axis='both', which='major', labelsize=15) 
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Comparison plot saved to: {output_path}")

def generate_all_comparison_plots():
    """Main function to generate both comparison plots."""
    
    # --- 1. I.I.D. Data Comparison ---
    print("\n--- Generating I.I.D. Comparison Plot ---")
    iid_files = [
        f'bound_dd_bounds_iid_N{T}.npz',
        f'bound_bootstrap_iid_N{T}.npz',
        f'bound_set_membership_iid_N{T}.npz'
    ]
    iid_bounds = load_bound_data(iid_files)
    if len(iid_bounds) == 3:
        plot_combined_bounds(
            bounds_list=iid_bounds,
            true_params=tuple(TRUE_PARAMS.values()),
            title=None,
            output_path=os.path.join(FIGURES_DIR, 'comparison_iid.pdf')
        )
    else:
        print("Could not generate I.I.D. plot because not all data files were found.")

    # --- 2. Trajectory Data Comparison ---
    print("\n--- Generating Trajectory Comparison Plot ---")
    trajectory_files = [
        f'bound_tsiams_trajectory_N{T}.npz',
        f'bound_bootstrap_trajectory_N{T}.npz',
        f'bound_set_membership_trajectory_N{T}.npz'
    ]
    trajectory_bounds = load_bound_data(trajectory_files)
    if len(trajectory_bounds) == 3:
        plot_combined_bounds(
            bounds_list=trajectory_bounds,
            true_params=tuple(TRUE_PARAMS.values()),
            title= None,
            output_path=os.path.join(FIGURES_DIR, 'comparison_trajectory.pdf')
        )
    else:
        print("Could not generate Trajectory plot because not all data files were found.")


if __name__ == '__main__':
    generate_all_comparison_plots()

