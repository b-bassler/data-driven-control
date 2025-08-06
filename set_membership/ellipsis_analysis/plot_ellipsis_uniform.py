import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# --- 1. Vorbereitung ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(9, 9))

# Farben für die verschiedenen Ellipsen
farben = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#8256ab", '#8c564b']

# --- 2. Daten aus dem Unterordner laden ---
# Name des Unterordners, in dem die Ergebnisse liegen
ordner_name = "results"

# Suchmuster, um alle .npz-Dateien im Unterordner zu finden
such_muster = os.path.join(ordner_name, 'simulation_ergebnis_*.npz')
dateiliste = sorted(glob.glob(such_muster)) # Sortieren für eine konsistente Reihenfolge

# Prüfen, ob Dateien gefunden wurden
if not dateiliste:
    print(f"Keine Ergebnis-Dateien im Ordner '{ordner_name}' gefunden!")
    # Beendet das Skript, wenn keine Dateien da sind
    exit()
    





labels = ["VanWaarde, T = 50", "VanWaarde, T = 100","Berberich T = 50", "Berberich T = 100", "without noise","without noise2", "without noise3", "stochastic noise", "stochastic max"]



# --- 3. Schleife zum Plotten jeder Ellipse ---

for i, dateiname in enumerate(dateiliste):
    # Lade die Daten aus der Datei
    with np.load(dateiname) as data:
        zentrum = data['zentrum']
        halbachsen = data['halbachsen']
        winkel_grad = data['winkel_grad']
    
    # Rekonstruiere die Plot-Punkte der Ellipse
    winkel_rad = np.radians(winkel_grad)
    t = np.linspace(0, 2 * np.pi, 200) # Mehr Punkte für eine glattere Kurve
    a, b = halbachsen[0], halbachsen[1] 
    
    x_punkte = (zentrum[0] 
                + a * np.cos(t) * np.cos(winkel_rad) 
                - b * np.sin(t) * np.sin(winkel_rad))
    y_punkte = (zentrum[1] 
                + a * np.cos(t) * np.sin(winkel_rad) 
                + b * np.sin(t) * np.cos(winkel_rad))
    
    # Zeichne die Ellipse mit passender Farbe und Beschriftung
    ax.plot(x_punkte, y_punkte, 
            color=farben[i % len(farben)], 
            linewidth=2,
            label=labels[i])




ax.set_aspect('equal', adjustable='box')
ax.legend(fontsize=11)
ax.set_xlabel(r'$A$', fontsize=14)
ax.set_ylabel(r'$B$', fontsize=14)
#ax.set_title('VanWaarde ', fontsize=16, pad=20)
plt.scatter(0.5, 0.5, marker='x', color='red', s=100, linewidths=2)
plt.tight_layout() 
plt.show()