import numpy as np
import matplotlib.pyplot as plt
import glob
import os


plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(9, 9))


farben = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#8256ab", '#8c564b']



ordner_name = "results"


such_muster = os.path.join(ordner_name, 'simulation_ergebnis_*.npz')
dateiliste = sorted(glob.glob(such_muster)) 

if not dateiliste:
    print(f"Keine Ergebnis-Dateien im Ordner '{ordner_name}' gefunden!")
    exit()
    
print(f"{len(dateiliste)} Ergebnis-Dateien gefunden. Beginne mit dem Plotten...")




labels = ["Uniform", "normally distributed, same variance","normally distributed, same interval", "Seed 4", "without noise","without noise2", "without noise3"]





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




ax.set_aspect('equal', adjustable='box')
ax.legend(fontsize=11)
ax.set_xlabel(r'$A$', fontsize=14)
ax.set_ylabel(r'$B$', fontsize=14)
ax.set_title('set-membership uniform noise ', fontsize=16, pad=20)
#plt.scatter(0.3, 0.8, c="red", marker="x", s=100, label="(A=0.3, B=0.8)", zorder=2)
plt.tight_layout() 
plt.show()
