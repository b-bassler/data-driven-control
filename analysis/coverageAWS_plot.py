import pandas as pd
import matplotlib.pyplot as plt
from typing import List


try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "text.latex.preamble": r"""
            \usepackage[T1]{fontenc}
            \usepackage[light]{firasans}
            \usepackage{amsmath}
        """,
    })
except:
    print("LaTeX ist nicht installiert. Nutze Standard-Schriftarten.")

def plot_coverage_meta_comparison(
    dataframe: pd.DataFrame,
    plot_configs: List,
    target_rate: float,
    output_path: str
    ) -> None:
    """
    Stellt die mittleren Fehlerraten aus einem Meta-Experiment für mehrere Methoden dar,
    jeweils mit einem schattierten Bereich, der +/- eine Standardabweichung darstellt.
    """
    fig, ax = plt.subplots(figsize=(12, 8)) 
    

    LABEL_FONTSIZE = 20      
    LEGEND_FONTSIZE = 18    
    TICK_FONTSIZE = 15      

  
    ax.axhline(y=target_rate, color='black', linestyle='--',
               label=f'Target Rate (5\%)')


    for config in plot_configs:
        t_values = dataframe['T']
        mean_rates = dataframe[config['mean_col']]
        std_devs = dataframe[config['std_dev_col']]
        
        ax.plot(t_values, mean_rates, marker=config['marker'], linestyle='-',
                label=config['label'], color=config['color'])

        ax.fill_between(t_values, mean_rates - std_devs, mean_rates + std_devs,
                        color=config['color'], alpha=0.2)
        

    

    ax.set_xlabel("Number of data points T", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Empiricl faillure rate", fontsize=LABEL_FONTSIZE)
    

    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    

    ax.legend(fontsize=LEGEND_FONTSIZE)
    

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{(y * 100):.0f}\%'))

    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"-> Plot gespeichert unter: {output_path}")



plot_configurations = [
    {
        'mean_col': 'bootstrap_mean_rate',
        'std_dev_col': 'bootstrap_std_dev_rate',
        'label': 'Bootstrap',
        'color': 'red',
        'marker': 'x'
    },
    {
        'mean_col': 'set_membership_mean_rate',
        'std_dev_col': 'set_membership_std_dev_rate',
        'label': 'Set Membership',
        'color': 'green',
        'marker': 's'
    }
]

csv_file = 'coverage_meta_analysis_summary.csv'
output_file = "coverage_plot_large_font.pdf"
target_failure_rate = 0.05

try:
    df = pd.read_csv(csv_file)
    plot_coverage_meta_comparison(
        dataframe=df,
        plot_configs=plot_configurations,
        target_rate=target_failure_rate,
        output_path=output_file
    )
except FileNotFoundError:
    print(f"FEHLER: Die Datei '{csv_file}' wurde nicht gefunden.")
    print("Stelle sicher, dass die CSV-Datei im selben Ordner wie das Skript liegt.")