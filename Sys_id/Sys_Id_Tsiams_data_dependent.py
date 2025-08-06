import numpy as np
import os   
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import math 

data_folder = r"C:\Users\benno\Desktop\Simulations\Data\generated_data"

daten_dict = {
    f: np.load(os.path.join(data_folder, f))
    for f in os.listdir(data_folder)
    if f.endswith(".npy")
}

state_data = daten_dict["state_data_gauss2.npy"]
input_data = daten_dict["input_data_gauss2.npy"]
sigma_u = daten_dict["sigma_u2.npy"]
sigma_w = daten_dict["sigma_w2.npy"]
sigma_w_quadrat = sigma_w**2
sigma_u_quadrat = sigma_u**2
A = daten_dict["A_Matrix.npy"]
B = daten_dict["B_Matrix.npy"]

print("sigma_w_quadrat:", sigma_w_quadrat)



T = 100 #trajectory length
N = 1 #fixed at 1

X = state_data[0:1, 0:T]
U = input_data[0:1, 0:T]

X_ges_plus = state_data[:N, 1:T+1]  # von x1 bis xT
X_ges_minus = state_data[:N, 0:T]   # von x0 bis xT-1
U_ges_minus = input_data[:N, 0:T]   # von x0 bis xT-1


X_N = X_ges_plus.reshape(-1, 1)
X_N_minus = X_ges_minus.reshape(-1, 1)
U_N_minus = U_ges_minus.reshape(-1, 1)



Z = np.block([[X_N_minus, U_N_minus]])               
 
AB_est = np.linalg.inv(Z.T @ Z) @ Z.T @ X_N

A_est = AB_est[:1, :]
B_est = AB_est[1:2, :]
print("Estimated A matrix:\n", A_est)
print("Estimated B matrix:\n", B_est)



delta = 0.05   # Lies within Radius by chance of 1-delta

V_t = np.array([
    [np.vdot(X, X), np.vdot(X, U)],
    [np.vdot(U, X), np.vdot(U, U)]
    ])


#Covariance of the state
T_t = 0
M = sigma_u_quadrat * (B @ B.T) + sigma_w_quadrat # Konstanter Term σ_u² B Bᵀ + Σ_w
t = 2 #define t, für tau = 2 und index t = tau/2 = 2

for k in range(t):
    Ak = np.linalg.matrix_power(A,k)
    T_t += Ak @ M @ Ak.T 

n = T_t.shape[0]
zero = np.zeros((n, n))

T_t_dach = np.block([[T_t,      zero],
                     [zero,     sigma_u_quadrat * np.eye(1)  ]]) 

print(T_t_dach)

c = 0.00140625
tau = 2

V = c * tau * math.floor(T/tau) * T_t_dach

V_dach = V_t + V

dx = X.shape[0]





# radius = 8 * sigma_w_quadrat * np.log(np.sqrt(np.linalg.det(V_dach)) * 5**dx /
#                                       np.sqrt(np.linalg.det(V)) * delta
#                                       ) * np.linalg.norm(sqrtm(V_dach) @ np.linalg.inv(sqrtm(V_t)), ord = 2)


log_term = np.log(
    (np.sqrt(np.linalg.det(V_dach)) / np.sqrt(np.linalg.det(V))) * (5**dx / delta)
)
norm_term_sq = np.linalg.norm(sqrtm(V_dach) @ np.linalg.inv(sqrtm(V_t)), ord=2)**2

radius = 8 * sigma_w_quadrat * log_term * norm_term_sq

print("Radius:", radius)



P = V_dach/radius   #charakteristische Matrix der Ellipse


A_est = A_est.item()
B_est = B_est.item()
print("A:",A_est)
print("B:", B_est)

# --- 2. BERECHNUNG DER ELLIPSEN-EIGENSCHAFTEN ---

# Eigenwerte und Eigenvektoren der Matrix P berechnen
eigenwerte, eigenvektoren = np.linalg.eig(P)
lambda1, lambda2 = eigenwerte

# Halbachsenlängen aus den Eigenwerten bestimmen (L = 1 / sqrt(lambda))
L1 = 1 / np.sqrt(lambda1)
L2 = 1 / np.sqrt(lambda2)

# Drehwinkel aus dem ersten Eigenvektor bestimmen
e1 = eigenvektoren[:, 0]
alpha = np.arctan2(e1[1], e1[0])

# Flächeninhalt der Ellipse
area = np.pi * L1 * L2


# Rand der Ellipse
t = np.linspace(0, 2 * np.pi, 100)
x_punkte = A_est + (L1 * np.cos(t) * np.cos(alpha) - L2 * np.sin(t) * np.sin(alpha))
y_punkte = B_est + (L1 * np.cos(t) * np.sin(alpha) + L2 * np.sin(t) * np.cos(alpha))


plt.figure(figsize=(10, 8))
plt.plot(x_punkte, y_punkte, label='Confidence-ellipse', color='blue')
plt.scatter(A_est, B_est, color='red',marker='x', zorder=5, label=f'Center ({A_est:.2f}, {B_est:.2f})')
ax = plt.gca()
ax.tick_params(axis='both', labelsize=20)

plt.xlabel('A',  fontsize=25)
plt.ylabel('B',  fontsize=25)
plt.legend()
plt.grid(True)
plt.axis('equal') 

plt.show()

# --- 3. ZUWEISUNG DER VARIABLEN FÜR DIE SPEICHERUNG ---

# Zentrum der Ellipse als Liste oder Array definieren
zentrum = [A_est, B_est]

# Halbachsenlängen in einer Liste oder einem Array zusammenfassen
# Es ist sinnvoll, sie der Größe nach zu sortieren (größte zuerst)
halbachsen = sorted([L1, L2], reverse=True)

# Den Winkel von Radiant in Grad umrechnen
winkel_grad = np.degrees(alpha)

# Die Variable für den Flächeninhalt zuweisen
flaecheninhalt = area

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

































script_ordner = os.path.dirname(os.path.abspath(__file__))


basis_ordner = "area_analysis"
unter_ordner = "ellipse_data_tsiamis"
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


neue_spalte = np.array([[area], [T]])
data = np.hstack((data, neue_spalte))

sortier_indizes = np.argsort(data[1, :])
data = data[:, sortier_indizes]

np.savez(pfad, ellipse_area_data=data)

print("Daten erfolgreich gespeichert!")
print(f"Pfad: '{pfad}'")



areas = data[0, :]
rollouts = data[1, :]


plt.figure(figsize=(10, 6))
plt.plot(rollouts, areas, marker='o', linestyle='-', color='b')
plt.title('Verlauf der Area über die Anzahl der Rollouts')
plt.xlabel('Anzahl der Rollouts (N)')
plt.ylabel('Area')
plt.grid(True)
plt.tight_layout()

plt.show()
