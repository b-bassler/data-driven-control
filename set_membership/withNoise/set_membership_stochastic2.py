import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy.stats import chi2
import os


from rsome import ro
from rsome import cpt_solver as cpt
import rsome as rso

#Number of datasamples used (T+1)
T = 400
sample_size = 700


data_folder = r"C:\Users\benno\Desktop\Simulations\Data\generated_data"

daten_dict = {
    f: np.load(os.path.join(data_folder, f))
    for f in os.listdir(data_folder)
    if f.endswith(".npy")
}



X = daten_dict["state_data_gauss.npy"]
U = daten_dict["input_data_gauss.npy"]

U_minus = U[:, 0:T]
X_plus = X[:, 1:T+1]
X_minus = X[:, 0:T]

# U_minus = U[:, 0+12*T:13*T]
# X_plus = X[:, 1+12*T:13*T+1]
# X_minus = X[:, 0+12*T:13*T]







A_min = 0.49
A_max = 0.51
B_min = 0.49
B_max = 0.51



#Noisemodel
k       = 2          # Freiheitsgrade
delta   = 0.05      #95%-Konfidenzniveau = 1 - delta


c_delta = chi2.ppf(1-delta, df=k)   
print("c_delta:", c_delta)
w_max = 0.01  
sigma_quadrat = (w_max**2) / 3
sigma_w = np.sqrt(sigma_quadrat)


z = 1.96 #Z-Quantil für 1-delta = 0.95
# a = w_max
# sigma_w = a / z

# sigma_quadrat = sigma_w**2


sigma_x2 = np.mean(X**2)
SNR = 3 * sigma_x2 / (w_max**2)
SNR_dB = round(10 * np.log10(SNR), 2)
print("SNR:", SNR_dB, "dB")


Phi11 = sigma_quadrat * c_delta* np.eye(X.shape[0]) 
Phi12 = np.zeros((X.shape[0], X_minus.shape[1]))
Phi22 = -np.linalg.pinv(np.block([[X_minus], [U_minus]])) @ np.block([[X_minus], [U_minus]])
Phi21 = Phi12.T

Phi = np.block([[Phi11, Phi12],
                 [Phi21, Phi22]])




A_vector = np.linspace(A_min, A_max, sample_size)
B_vector = np.linspace(B_min, B_max, sample_size)
#print("A_vector:", A_vector)
#print("B_vector:", B_vector)    
A_mesh, B_mesh = np.meshgrid(A_vector, B_vector)

# Liste für gültige Paare
valid_pairs = []


for i in range(A_mesh.shape[0]):
    for j in range(A_mesh.shape[1]):
        A_val = A_mesh[i, j]
        B_val = B_mesh[i, j]

        W = X_plus - A_val * X_minus - B_val * U_minus
        
        eigs_matrix = np.block([[np.eye(X_minus.shape[0])], [W.T] ]).T   @   Phi   @   np.block([[np.eye(X_minus.shape[0])], [W.T] ])

        eigs = np.linalg.eigvals(eigs_matrix)

        if np.all(eigs > 0):
            valid_pairs.append((A_val, B_val))

print(f"Anzahl der (A,B)-Paare mit allen Eigenwerten > 0: {len(valid_pairs)}")


#if valid_pairs:
#    A_ok, B_ok = zip(*valid_pairs)
#    plt.figure()
#    plt.scatter(0.3, 0.8, c="red", marker="x", s=100, label="(A=0.3, B=0.8)", zorder=2)
#    plt.scatter(A_ok, B_ok, s=15)
#    #plt.xlim(0.25, 0.35)   
#    #plt.ylim(0.75, 0.85)
#    plt.xlabel("A")
#    plt.ylabel("B")
#    plt.title(f"Matching (A,B), with SNR of {SNR_dB} in dB")
#    plt.grid(True)
#    plt.show()
#    plt.scatter(A_ok, B_ok, s=15, label="True values")
   
#else:
#    print("Keine (A,B)-Paare mit allen Eigenwerten > 0 gefunden.")



pair_array = np.array(valid_pairs, dtype=np.float64)


m = pair_array.shape[0]

xs = pair_array
model = ro.Model()

P = model.dvar((2, 2))
c = model.dvar(2)
Z = rso.tril(model.dvar((2, 2)))
v = model.dvar(2)

model.max(v.sum())
model.st(v <= rso.log(rso.diag(Z)))
model.st(rso.rstack([P, Z], 
                    [Z.T, rso.diag(Z, fill=True)]) >> 0)
for i in range(m):
    model.st(rso.norm(P@xs[i] - c) <= 1)
model.st(P >> 0)

model.solve(cpt)
print(f'Determinant: {np.exp(model.get())}')

Ps = P.get()
cs = c.get()

step = 0.01
t = np.arange(0, 2*np.pi+step, step)
y = np.vstack((np.cos(t), np.sin(t))).T

ellip = np.linalg.inv(Ps) @ (y + cs).T

plt.figure(figsize=(5, 5))
plt.scatter(xs[:, 0], xs[:, 1], 
             facecolor='none', marker = ".", color='k', label='Data points')
plt.plot(ellip[0], ellip[1], color='b', 
         label='Minimum enclosing ellipsoid')
plt.legend(fontsize=12, bbox_to_anchor=(1.01, 1.02))
plt.axis('equal')
plt.xlabel(r'$A$', fontsize=14)
plt.ylabel(r'$B$', fontsize=14)
plt.show()

# Zentrum der Ellipse 
zentrum = np.linalg.inv(Ps) @ cs

# Halbachsen und Rotation 
A_shape = Ps.T @ Ps
eigenwerte, eigenvektoren = np.linalg.eig(A_shape)

# Länge der Halbachsen
sort_indices = np.argsort(eigenwerte)[::-1]
eigenwerte = eigenwerte[sort_indices]
eigenvektoren = eigenvektoren[:, sort_indices]


halbachsen = 1 / np.sqrt(eigenwerte)

# Rotationswinkel 
winkel_rad = np.arctan2(eigenvektoren[1, 0], eigenvektoren[0, 0])
winkel_grad = np.degrees(winkel_rad)

# Flächeninhalt 
flaecheninhalt = np.pi * halbachsen[0] * halbachsen[1]


print("\n--- Ellipsen-Eigenschaften ---")
print(f"Zentrum (x, y): ({zentrum[0]:.5f}, {zentrum[1]:.5f})")
print(f"Länge der Halbachsen: {halbachsen[0]:.5f} und {halbachsen[1]:.5f}")
print(f"Rotation der Hauptachse: {winkel_grad:.5f}°")
print(f"Flächeninhalt: {flaecheninhalt:.7f}")
print("----------------------------\n")



ordner_name = "results"


os.makedirs(ordner_name, exist_ok=True)


dateiname = "simulation_ergebnis_1.npz"
voller_pfad = os.path.join(ordner_name, dateiname)


np.savez(voller_pfad, 
          zentrum=zentrum, 
           halbachsen=halbachsen, 
            winkel_grad=winkel_grad)

print(f"Ergebnisse in '{voller_pfad}' gespeichert.")


#
#---------------------------------------------------------------------------------------------------------------------------
#

#----------------------------------------------
# Bestimmung von max_a, max_b und Worst-Case
#----------------------------------------------

# 1. Generiere Punkte der Ellipse relativ zum Zentrum (0,0)
# Wir brauchen genügend Punkte für eine gute Genauigkeit
phi = np.linspace(0, 2 * np.pi, 1000) 
circle_points = np.vstack([np.cos(phi), np.sin(phi)])

# 2. Transformationsmatrix aus den sortierten Werten erstellen und anwenden
# eigenvektoren sind bereits sortiert, halbachsen auch.
# halbachsen[0] ist die kurze, halbachsen[1] die lange Achse.
ellipse_transform = eigenvektoren @ np.diag(halbachsen)
ellipse_points_centered = ellipse_transform @ circle_points

# 3. Finde die maximale absolute Koordinate entlang jeder Achse
# Das entspricht der halben Breite/Höhe der Bounding Box der Ellipse.
max_a = np.max(np.abs(ellipse_points_centered[0, :]))
max_b = np.max(np.abs(ellipse_points_centered[1, :]))

# 4. Die "Worst-Case Abweichung" ist einfach die längste Halbachse.
# Da die "halbachsen" sortiert sind (von kurz nach lang), ist es der letzte Wert.
worst_case_deviation = halbachsen[-1]


print("\n--- Abgeleitete Metriken für die Speicherung ---")
print(f"Maximale Ausdehnung in a (x-Richtung der Bounding Box): {max_a:.4f}")
print(f"Maximale Ausdehnung in b (y-Richtung der Bounding Box): {max_b:.4f}")
print(f"Worst-Case Abweichung (längste Halbachse): {worst_case_deviation:.4f}")
print("------------------------------------------------\n")


#----------------------------------------------
#delta Vergleich (Speichern der Daten)
#----------------------------------------------

script_ordner = os.path.dirname(os.path.abspath(__file__))


basis_ordner = "delta_ab"
unter_ordner = "ellipse_data_uniform"
voller_ordner = os.path.join(script_ordner, basis_ordner, unter_ordner)

os.makedirs(voller_ordner, exist_ok=True) 


pfad = os.path.join(voller_ordner, "set_memb_abmax_data.npz")


if os.path.exists(pfad):
    try:
        
        data = np.load(pfad)["set_memb_abmax_data"]
    except (IOError, KeyError):
        print(f"Warnung: Datei '{pfad}' war fehlerhaft. Es wird ein neues leeres Datenarray erstellt.")
        data = np.empty((4, 0))
else:
    
    data = np.empty((4, 0))


# Der Wert T muss aus dem oberen Teil deines Skripts bekannt sein
# Falls T nicht existiert, ersetze es hier durch die korrekte Variable
neue_spalte = np.array([[max_a], [max_b], [worst_case_deviation], [T]])
data = np.hstack((data, neue_spalte))

sortier_indizes = np.argsort(data[3, :])
data = data[:, sortier_indizes]
np.savez(pfad, set_memb_abmax_data=data)

print("Daten erfolgreich gespeichert!")
print(f"Pfad: '{pfad}'")
print("Aktuelle Daten im Array:\n", data)






area = flaecheninhalt
N    = T

script_ordner = os.path.dirname(os.path.abspath(__file__))


basis_ordner = "area_analysis"
unter_ordner = "ellipse_data"
voller_ordner = os.path.join(script_ordner, basis_ordner, unter_ordner)

os.makedirs(voller_ordner, exist_ok=True)


pfad = os.path.join(voller_ordner, "ellipse_area_data.npz")


if os.path.exists(pfad):
    try:
        data = np.load(pfad)["ellipse_area_data"]
    except (IOError, KeyError):
        print(f"Warnung: Datei '{pfad}' war fehlerhaft. Es wird ein neues leeres Datenarray erstellt.")
        data = np.empty((2, 0))
else:
    data = np.empty((2, 0))


neue_spalte = np.array([[area], [N]])
data = np.hstack((data, neue_spalte))

sortier_indizes = np.argsort(data[1, :])
data = data[:, sortier_indizes]

np.savez(pfad, ellipse_area_data=data)

print("Daten erfolgreich gespeichert!")
print(f"Pfad: '{pfad}'")
print("Aktuelle Daten im Array:\n", data)


areas = data[0, :]
rollouts = data[1, :]


plt.figure(figsize=(10, 6))
plt.plot(rollouts, areas, marker='o', linestyle='-', color='b')
plt.title('Verlauf der Area über die Anzahl der Datenpunkte')
plt.xlabel('Anzahl der Datenpunkte (N)')
plt.ylabel('Area')
plt.grid(True)
plt.tight_layout()


plot_pfad = os.path.join(voller_ordner, 'area_verlauf.png')
plt.savefig(plot_pfad)

print(f"\nPlot wurde erfolgreich als '{plot_pfad}' gespeichert.")