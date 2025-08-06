import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import os

from rsome import ro
from rsome import cpt_solver as cpt
import rsome as rso


#Better and tighter results as VanWaarde for uniform Noise
#Number of datasamples used (T+1)
T = 100
data_folder = r"C:\Users\benno\Desktop\Simulations\Data\generated_data"
daten_dict = {
    f: np.load(os.path.join(data_folder, f))
    for f in os.listdir(data_folder)
    if f.endswith(".npy")
}


X = daten_dict["state_data_uniform.npy"]
U = daten_dict["input_data_uniform.npy"]
W = daten_dict["noise_data_uniform.npy"]

X_plus = X[:, 1:T+1]
X_minus = X[:, 0:T]
U_minus = U[:, 0:T]
#print("Shape of X_plus:", X_plus.shape)
#print("Shape of X_minus:", X_minus.shape)
#print("Shape of U_minus:", U_minus.shape)


A_min = 0.45
A_max = 0.55
B_min = 0.45
B_max = 0.55

sample_size = 100

#Nosemodel, Random uniform noise with maximum amplitude w_max
w_max = 0.01
sigma_x2 = np.mean(X**2)
print("Mean of X^2:", sigma_x2)

SNR = 3 * sigma_x2 / (w_max**2)
SNR_dB = round(10 * np.log10(SNR), 2)
print("SNR:", SNR_dB, "dB")

Qw = -np.eye(X.shape[0]) 
Sw = np.zeros((X.shape[0], X_minus.shape[1]))
Rw = T*w_max**2 * np.eye(X_minus.shape[1])    #korrigierter Block mit T*w_max^2

Phi = np.block([[Qw, Sw],
                 [Sw.T, Rw]])




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
        
        eigs_matrix = np.block([[W], [np.eye(T)]]).T   @   Phi   @   np.block([[W], [np.eye(T)]])

        eigs = np.linalg.eigvals(eigs_matrix)

        if np.all(eigs > 0):
            valid_pairs.append((A_val, B_val))

print(f"Anzahl der (A,B)-Paare mit allen Eigenwerten > 0: {len(valid_pairs)}")


#if valid_pairs:
 #   A_ok, B_ok = zip(*valid_pairs)
  #  plt.figure()
   # plt.scatter(0.3, 0.8, c="red", marker="x", s=100, label="(A=0.3, B=0.8)", zorder=2)
    #plt.scatter(A_ok, B_ok, s=15)
   # plt.xlim(0.25, 0.35)   
   # plt.ylim(0.75, 0.85)
   # plt.xlabel("A")
    #plt.ylabel("B")
    #plt.title(f"Matching (A,B), with SNR of {SNR_dB} in dB")
    #plt.grid(True)
    #plt.show()
    #plt.scatter(A_ok, B_ok, s=15, label="True values")
   
#else:
 #   print("Keine (A,B)-Paare mit allen Eigenwerten > 0 gefunden.")


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
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.show()

# 1. Das Zentrum der Ellipse berechnen
# Das Zentrum ist nicht direkt 'cs'. Man muss es aus P und c zurückrechnen.
zentrum = np.linalg.inv(Ps) @ cs

# 2. Halbachsen und Rotation aus der korrekten Matrix A = Ps.T @ Ps berechnen
A_shape = Ps.T @ Ps
eigenwerte, eigenvektoren = np.linalg.eig(A_shape)

# Die Länge der Halbachsen ist 1 / sqrt(Eigenwerte)
# Wir sortieren die Eigenwerte, um eine konsistente Reihenfolge zu gewährleisten
sort_indices = np.argsort(eigenwerte)[::-1]
eigenwerte = eigenwerte[sort_indices]
eigenvektoren = eigenvektoren[:, sort_indices]
# Die Länge der Halbachsen ist die Wurzel der Eigenwerte
halbachsen = 1 / np.sqrt(eigenwerte)

# Der Rotationswinkel ergibt sich aus dem ersten Eigenvektor
# Wir verwenden arctan2 für numerische Stabilität
winkel_rad = np.arctan2(eigenvektoren[1, 0], eigenvektoren[0, 0])
winkel_grad = np.degrees(winkel_rad)

# Flächeninhalt berechnen 
flaecheninhalt = np.pi * halbachsen[0] * halbachsen[1]

# 3. Die Ergebnisse ausgeben
print("\n--- Ellipsen-Eigenschaften ---")
print(f"Zentrum (x, y): ({zentrum[0]:.5f}, {zentrum[1]:.5f})")
print(f"Länge der Halbachsen: {halbachsen[0]:.5f} und {halbachsen[1]:.5f}")
print(f"Rotation der Hauptachse: {winkel_grad:.5f}°")
print(f"Flächeninhalt: {flaecheninhalt:.7f}")
print("----------------------------\n")


# 1. Namen des Unterordners definieren
ordner_name = "results"

# 2. Sicherstellen, dass der Ordner existiert (erstellt ihn, falls nicht)
os.makedirs(ordner_name, exist_ok=True)

# 3. Dateinamen und vollständigen Pfad erstellen
dateiname = "simulation_ergebnis_4.npz"
voller_pfad = os.path.join(ordner_name, dateiname)

np.savez(voller_pfad, 
          zentrum=zentrum, 
           halbachsen=halbachsen, 
            winkel_grad=winkel_grad)

print(f"Ergebnisse in '{voller_pfad}' gespeichert.")