import numpy as np
import matplotlib.pyplot as plt
import os

pfad_rechteck = r"C:\Users\benno\Desktop\Simulations\Sys_id\area_analysis\rectangle_data\rectangle_area_data.npz"
pfad_ellipse = r"C:\Users\benno\Desktop\Simulations\set_membership\withNoise\area_analysis\ellipse_data\ellipse_area_data.npz"
pfad_ellipse_tsiamis = r"C:\Users\benno\Desktop\Simulations\Sys_id\area_analysis\ellipse_data_tsiamis\ellipse_area_data.npz"


try:
    data_rechteck = np.load(pfad_rechteck)["rectangle_area_data"]
    data_ellipse = np.load(pfad_ellipse)["ellipse_area_data"]
    data_ellipse = data_ellipse[:,1:]
    data_ellipse_tsiamis = np.load(pfad_ellipse_tsiamis)["ellipse_area_data"]
    print("Alle Datensätze erfolgreich geladen! ✅")
except FileNotFoundError as e:
    print(f"\nFehler: Datei nicht gefunden! Stelle sicher, dass die Ordnerstruktur korrekt ist.")
    print(f"Fehlender Pfad: {e.filename}")
    exit()


# 3.1 Rechteck-Daten sortieren
sortier_indizes_rechteck = np.argsort(data_rechteck[1, :])
sorted_rechteck_n = data_rechteck[1, sortier_indizes_rechteck]
sorted_rechteck_area = data_rechteck[0, sortier_indizes_rechteck]

# 3.2 Ellipsen-Daten sortieren
sortier_indizes_ellipse = np.argsort(data_ellipse[1, :])
sorted_ellipse_n = data_ellipse[1, sortier_indizes_ellipse]
sorted_ellipse_area = data_ellipse[0, sortier_indizes_ellipse]

sortier_indizes_ellipse_tsiamis = np.argsort(data_ellipse_tsiamis[1, :])
sorted_ellipse_n_tsiamis = data_ellipse_tsiamis[1, sortier_indizes_ellipse_tsiamis]
sorted_ellipse_area_tsiamis = data_ellipse_tsiamis[0, sortier_indizes_ellipse_tsiamis]


length = 300



# 4. Diagramm erstellen und die *sortierten* Daten plotten
plt.figure(figsize=(12, 7))

# Plot für Rechteck-Daten
plt.plot(
    sorted_rechteck_n[ 0:length],      
    sorted_rechteck_area[0:length],   
    marker='o',
    linestyle='-',
    label='system identification bootstrap bound ',
    markersize=2
)


plt.plot(
    sorted_ellipse_n[0:length],       
    sorted_ellipse_area[ 0:length],    
    marker='s',
    linestyle='--',
    label='set membership',
    markersize = 2
)

plt.plot(
    sorted_ellipse_n_tsiamis[0:length],       
    sorted_ellipse_area_tsiamis[0:length],    
    marker='s',
    linestyle='--',
    label='system identification data dependent bounds',
    markersize = 2
)



plt.xlabel('Number of data points used')
plt.ylabel('Area')
plt.legend(fontsize = 15)
plt.grid(True)
plt.tight_layout()



plt.show()