import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.patches import Rectangle 

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['savefig.bbox']      = 'tight'
plt.rcParams['savefig.pad_inches']= 0

fig, ax = plt.subplots(figsize=(9, 9))

farben = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#8256ab", '#8c564b']

#======================================================================
# --- Manuelles Rechteck hinzufügen ---
#======================================================================
rect_center_A = 0.300578
rect_center_B = 0.800649
rect_error_A = 0.002047
rect_error_B = 0.001502

manual_rect = Rectangle(
    (rect_center_A - rect_error_A, rect_center_B - rect_error_B),
    2 * rect_error_A,
    2 * rect_error_B,
    edgecolor='purple',
    facecolor='purple',
    alpha=0.2,
    label='Bootstrap uncertainty'
)
ax.add_patch(manual_rect)

true_A = 0.3
true_B = 0.8
ax.plot(true_A, true_B, 
        marker='x', color='green', markersize=8, 
        linestyle='None', label='true A and B')

#======================================================================
# --- Ellipsen aus Dateien plotten & Grenzen sammeln ---
#======================================================================
# Initialisieren der Gesamtgrenzen mit den Werten des Rechtecks
total_x_min = rect_center_A - rect_error_A
total_x_max = rect_center_A + rect_error_A
total_y_min = rect_center_B - rect_error_B
total_y_max = rect_center_B + rect_error_B

ordner_name = "results"
such_muster = os.path.join(ordner_name, 'simulation_ergebnis_*.npz')
dateiliste = sorted(glob.glob(such_muster))

if not dateiliste:
    print(f"Keine Ergebnis-Dateien im Ordner '{ordner_name}' gefunden!")
else:
    print(f"{len(dateiliste)} Ergebnis-Dateien gefunden. Beginne mit dem Plotten...")
    labels = ["Seed 1", "Seed 2","Seed 3", "Seed 4", "without noise","without noise2", "without noise3"]

    for i, dateiname in enumerate(dateiliste):
        with np.load(dateiname) as data:
            zentrum = data['zentrum']
            halbachsen = data['halbachsen']
            winkel_grad = data['winkel_grad']
        
        winkel_rad = np.radians(winkel_grad)
        t = np.linspace(0, 2 * np.pi, 200)
        a, b = halbachsen[0], halbachsen[1]
        
        x_punkte = (zentrum[0]
                    + a * np.cos(t) * np.cos(winkel_rad)
                    - b * np.sin(t) * np.sin(winkel_rad))
        y_punkte = (zentrum[1]
                    + a * np.cos(t) * np.sin(winkel_rad)
                    + b * np.sin(t) * np.cos(winkel_rad))
        
        ax.plot(x_punkte, y_punkte,
                color=farben[i % len(farben)],
                linewidth=2,
                label=labels[i])
        
        # Gesamtgrenzen bei jeder Ellipse aktualisieren
        total_x_min = min(total_x_min, np.min(x_punkte))
        total_x_max = max(total_x_max, np.max(x_punkte))
        total_y_min = min(total_y_min, np.min(y_punkte))
        total_y_max = max(total_y_max, np.max(y_punkte))

#======================================================================
# --- Finale Plot-Einstellungen mit Padding ---
#======================================================================

padding_factor = 0.2 

x_range = total_x_max - total_x_min
y_range = total_y_max - total_y_min

ax.set_xlim(total_x_min - x_range * padding_factor, 
            total_x_max + x_range * padding_factor)
ax.set_ylim(total_y_min - y_range * padding_factor, 
            total_y_max + y_range * padding_factor)

ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_aspect('equal', adjustable='box')


ax.legend(loc='upper right', fontsize=11) 
ax.set_xlabel(r'$A$', fontsize=16)
ax.set_ylabel(r'$B$', fontsize=16)
ax.set_title('Comparison set membership and system identification', fontsize=16, pad=20)

plt.tight_layout()
plt.show()
