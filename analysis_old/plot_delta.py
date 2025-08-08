
#erste Zeile: a_max
#zweite Zeile: b_max
#dritte Zeile: max_abweichung
#vierte Zeile: Anzahl Datenpunkte N 


import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Daten laden ---
# Pfade zu den gespeicherten .npz-Dateien
pfad_dean_bootstrap = r"C:\Users\benno\Desktop\Simulations\Sys_id\delta_ab\rectangle_data\rectangle_abmax_data.npz"
pfad_dean_ellipse = r"C:\Users\benno\Desktop\Simulations\Sys_id\delta_ab\ellipse_data\ellipse_dean_abmax_data.npz"
pfad_VanWaarde_ellipse = r"C:\Users\benno\Desktop\Simulations\set_membership\withNoise\delta_ab\ellipse_data_uniform\set_memb_abmax_data.npz"


# Überprüfen, ob die Dateien existieren
if not os.path.exists(pfad_dean_bootstrap) or not os.path.exists(pfad_dean_ellipse):
    print("Fehler: Mindestens eine der Datendateien wurde nicht gefunden.")
    print("Bitte überprüfen Sie die Pfade:")
    print(f"- {pfad_dean_bootstrap}")
    print(f"- {pfad_dean_ellipse}")
    exit()

# Daten aus den .npz-Dateien laden
data_rect = np.load(pfad_dean_bootstrap)["rectangle_abmax_data"]
data_ellipse = np.load(pfad_dean_ellipse)["ellipse_dean_abmax_data"]
data_ellipse2 = np.load(pfad_VanWaarde_ellipse)["set_memb_abmax_data"]

# Daten für bessere Lesbarkeit extrahieren
# Rechteck-Methode (Bootstrap)
rect_a_max = data_rect[0, :]
rect_b_max = data_rect[1, :]
rect_worst_case = data_rect[2, :]
N_rect = data_rect[3, :] # Anzahl der Datenpunkte

# Ellipsen-Methode
ellipse_a_max = data_ellipse[0, :]
ellipse_b_max = data_ellipse[1, :]
ellipse_worst_case = data_ellipse[2, :]
N_ellipse = data_ellipse[3, :] # Anzahl der Datenpunkte

ellipse_a_max2 = data_ellipse2[0, :]
ellipse_b_max2 = data_ellipse2[1, :]
ellipse_worst_case2 = data_ellipse2[2, :]
N_ellipse2 = data_ellipse2[3, :] # Anzahl der Datenpunkte


# --- 2. Diagramme erstellen ---

# Stil für die Plots setzen
plt.style.use('seaborn-v0_8-whitegrid')

# Diagramm 1: Achsenparallele Abweichungen
fig1, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(N_rect, rect_a_max, 'b-', label='Bootstrap max $|a - \hat{a}|$')
ax1.plot(N_rect, rect_b_max, 'b--', label='Bootstrap max $|b - \hat{b}|$')
ax1.plot(N_ellipse, ellipse_a_max, 'r-', label='SysId Bounds Dean max $|a - \hat{a}|$')
ax1.plot(N_ellipse, ellipse_b_max, 'r--', label='SysId Bounds Dean max $|b - \hat{b}|$')
ax1.plot(N_ellipse2, ellipse_a_max2, 'm-', label='VanWaarde set max $|a - \hat{a}|$') 
ax1.plot(N_ellipse2, ellipse_b_max2, 'm--', label='VanWaarde set max $|b - \hat{b}|$') 

ax1.set_title('Axis aligned deviation', fontsize=16)
ax1.set_xlabel('T', fontsize=12)
ax1.set_ylabel('max deviation', fontsize=12)
ax1.set_yscale('log') # Logarithmische Skala
ax1.legend(fontsize=11)
ax1.grid(True, which='both', linestyle=':')

# ---

# Diagramm 2: Worst-Case-Abweichung
fig2, ax2 = plt.subplots(figsize=(12, 8))

ax2.plot(N_rect, rect_worst_case, 'b-', label='Bootstrap $d_{max}$ (Ecke)')
ax2.plot(N_ellipse, ellipse_worst_case, 'r-', label='SysId Bounds Dean $d_{max}$ (lange Halbachse)')
ax2.plot(N_ellipse2, ellipse_worst_case2, '--', label='VanWaarde set $d_{max}$ (lange Halbachse)')

ax2.set_title('Worst-case deviation', fontsize=16)
ax2.set_xlabel('T', fontsize=12)
ax2.set_ylabel('max deviation ($d_{max}$)', fontsize=12)
ax2.set_yscale('log')
ax2.legend(fontsize=11)
ax2.grid(True, which='both', linestyle=':')

# --- 3. Alle Diagramme anzeigen ---
plt.tight_layout()
plt.show()