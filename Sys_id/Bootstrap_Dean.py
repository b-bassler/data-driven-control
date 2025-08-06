import numpy as np
import os
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from tqdm import tqdm

# ======================================================================
# NEUER PARAMETER: Anzahl der zu verwendenden Datenpunkte
# ======================================================================
# Lege hier fest, wie viele Zeitschritte (T) der geladenen Daten
# für die Analyse verwendet werden sollen.
# Setze den Wert auf None, um alle Daten zu verwenden.
T_datenpunkte_zu_verwenden = 400  # Beispiel: Nur die ersten 500 Zeitschritte verwenden
# ======================================================================

data_folder = r"C:\Users\benno\Desktop\Simulations\Data\generated_data"

daten_dict = {
    f: np.load(os.path.join(data_folder, f))
    for f in os.listdir(data_folder)
    if f.endswith(".npy")
}

real_state_data = daten_dict["state_data_gauss2.npy"]
real_input_data = daten_dict["input_data_gauss2.npy"]

# ======================================================================
# DATEN AUF GEWÜNSCHTE LÄNGE (T) ZUSCHNEIDEN (EINZIGE ÄNDERUNG)
# ======================================================================
if T_datenpunkte_zu_verwenden is not None:
    # Sicherstellen, dass nicht mehr Daten angefordert werden, als vorhanden sind
    max_T_verfügbar = real_input_data.shape[1]
    T_verwendet = min(T_datenpunkte_zu_verwenden, max_T_verfügbar)

    print(f"INFO: Verwende die ersten {T_verwendet} von {max_T_verfügbar} verfügbaren Zeitschritten.")

    # state_data hat T+1 Spalten, input_data hat T Spalten
    real_state_data = real_state_data[:, :T_verwendet + 1]
    real_input_data = real_input_data[:, :T_verwendet]
# ======================================================================

sigma_u = daten_dict["sigma_u2.npy"].item()
sigma_w = daten_dict["sigma_w2.npy"].item()

A_true = daten_dict["A_Matrix.npy"]
B_true = daten_dict["B_Matrix.npy"]


def least_squares_estimator(state_data, input_data):
    """
    Führt die Least-Squares-Schätzung für ein oder mehrere Rollouts durch.
    
    Args:
        state_data (np.array): Array der Zustandsdaten mit Shape (N, T+1).
        input_data (np.array): Array der Eingabedaten mit Shape (N, T).
        
    Returns:
        tuple: Ein Tupel mit (A_est, B_est).
    """
    # Anzahl der Rollouts und Länge aus den Daten ableiten
    N, T_plus_1 = state_data.shape
    T = T_plus_1 - 1

    # Matrizen aufbauen
    X_ges_plus = state_data[:N, 1:T+1]
    X_ges_minus = state_data[:N, 0:T]
    U_ges_minus = input_data[:N, 0:T]

    X_N = X_ges_plus.reshape(-1, 1)
    X_N_minus = X_ges_minus.reshape(-1, 1)
    U_N_minus = U_ges_minus.reshape(-1, 1)

    Z = np.block([[X_N_minus, U_N_minus]])

    # Schätzung durchführen (pinv ist robuster als inv)
    AB_est = np.linalg.pinv(Z.T @ Z) @ Z.T @ X_N

    A_est = AB_est[:1, :]
    B_est = AB_est[1:2, :]
    
    return A_est, B_est


#Erste Schätzung
A_dach, B_dach = least_squares_estimator(real_state_data, real_input_data)

print("Test,Test, A und B:", A_dach, B_dach)


def simuliere_system(A_sys, B_sys, T, N, sigma_u, sigma_w, x0=0):
    """
    Simuliert ein lineares System für N Rollouts über T Zeitschritte.

    Args:
        A_sys (np.array): Die Systemmatrix A.
        B_sys (np.array): Die Eingabematrix B.
        T (int): Anzahl der Zeitschritte pro Rollout.
        N (int): Anzahl der Rollouts (Versuchsreihen).
        sigma_u (float): Standardabweichung des Eingangssignals u.
        sigma_w (float): Standardabweichung des Prozessrauschens w.
        x0 (float, optional): Der Startzustand. Standard ist 0.

    Returns:
        tuple: Ein Tupel (x, u) mit den Zustands- und Eingabedaten.
               x hat die Form (N, T+1), u hat die Form (N, T).
    """
    mean = 0
    
    # Zufällige Eingangs- und Störsignale erzeugen
    u = np.random.normal(mean, sigma_u, (N, T))
    w = np.random.normal(mean, sigma_w, (N, T))
    
    # Zustandsvektor initialisieren
    x = np.zeros((N, T + 1))
    x[:, 0] = x0
    
    # System über die Zeit simulieren (vektorisierte Version)
    for k in range(T):
        x[:, k + 1] = A_sys * x[:, k] + B_sys * u[:, k] + w[:, k]
        
    return x, u


def berechne_norm_fehler(matrix1, matrix2):
    fehler = np.linalg.norm(matrix1 - matrix2, ord=2)
    return fehler


print("Berechne erste Schätzung (Anker)...")
A_dach, B_dach = least_squares_estimator(real_state_data, real_input_data)
print(f"Erste Schätzung: A_dach = {A_dach.item():.6f}, B_dach = {B_dach.item():.6f}")


#======================================================================
#Bootstrap-Schleife


# Parameter für den Bootstrap-Prozess
M = 2000 # Anzahl der Bootstrap-Versuche
delta = 0.05 # Konfidenzparameter
N_real, T_real_plus_1 = real_state_data.shape
T_real = T_real_plus_1 - 1


fehler_A_liste = []
fehler_B_liste = []

print(f"\nStarte Bootstrap-Prozess mit {M} Versuchen...")


for i in tqdm(range(M), desc="Bootstrap-Fortschritt"):

    # Synthetische Daten
    x_synthetisch, u_synthetisch = simuliere_system(A_dach, B_dach, T_real, N_real, sigma_u, sigma_w)

    # Neu schätzen mit den synthetischen Daten
    A_tilde, B_tilde = least_squares_estimator(x_synthetisch, u_synthetisch)

    # Fehler berechnen und speichern
    fehler_A = berechne_norm_fehler(A_dach, A_tilde)
    fehler_B = berechne_norm_fehler(B_dach, B_tilde)

    fehler_A_liste.append(fehler_A)
    fehler_B_liste.append(fehler_B)

print("Bootstrap-Prozess abgeschlossen.")

#Endergebnis aus den gesammelten Fehlern berechnen ---
epsilon_A_dach = np.percentile(fehler_A_liste, 100 * (1 - delta))
epsilon_B_dach = np.percentile(fehler_B_liste, 100 * (1 - delta))

print("\n--- BOOTSTRAP ERGEBNISSE ---")
print(f"Geschätzter Fehler für A (ε_A) mit {100*(1-delta)}% Konfidenz: {epsilon_A_dach:.6f}")
print(f"Geschätzter Fehler für B (ε_B) mit {100*(1-delta)}% Konfidenz: {epsilon_B_dach:.6f}")


fig, ax = plt.subplots()


rect = Rectangle(
    (A_dach.item() - epsilon_A_dach, B_dach.item() - epsilon_B_dach),
    2 * epsilon_A_dach,
    2 * epsilon_B_dach,
    edgecolor='black',
    facecolor='blue',
    alpha=0.3,
    label=f'{(1-delta)*100}% Confidence interval'
)
ax.add_patch(rect)


ax.plot(
    A_dach.item(),
    B_dach.item(),
    marker='x',
    color='red',
    linestyle='None',
    markersize=8,
    label='estimated (A, B)'
)


ax.plot(
    A_true.item(),
    B_true.item(),
    marker='x',
    color='green',
    linestyle='None',
    markersize=8,
    label='true (A, B)'
)


ax.set_xlabel("A")
ax.set_ylabel("B")
ax.set_title("Bootstrap model uncertainty")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)


x_min = A_dach.item() - 1.5 * epsilon_A_dach
x_max = A_dach.item() + 1.5 * epsilon_A_dach
y_min = B_dach.item() - 1.5 * epsilon_B_dach
y_max = B_dach.item() + 1.5 * epsilon_B_dach
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_aspect('equal', adjustable='box')

plt.show()


area = 4 * epsilon_A_dach * epsilon_B_dach
N    = N_real * T_real

script_ordner = os.path.dirname(os.path.abspath(__file__))


basis_ordner = "area_analysis"
unter_ordner = "rectangle_data"
voller_ordner = os.path.join(script_ordner, basis_ordner, unter_ordner)

os.makedirs(voller_ordner, exist_ok=True)


pfad = os.path.join(voller_ordner, "rectangle_area_data.npz")


if os.path.exists(pfad):
    try:
        
        data = np.load(pfad)["rectangle_area_data"]
    except (IOError, KeyError):
        print(f"Warnung: Datei '{pfad}' war fehlerhaft. Es wird ein neues leeres Datenarray erstellt.")
        data = np.empty((2, 0))
else:
    
    data = np.empty((2, 0))


neue_spalte = np.array([[area], [N]])
data = np.hstack((data, neue_spalte))

sortier_indizes = np.argsort(data[1, :])
data = data[:, sortier_indizes]

np.savez(pfad, rectangle_area_data=data)

print("Daten erfolgreich gespeichert!")
print(f"Pfad: '{pfad}'")
print("Aktuelle Daten im Array:\n", data)


areas = data[0, :]
rollouts = data[1, :]


plt.figure(figsize=(10, 6))
plt.plot(rollouts, areas, marker='o', linestyle='-', color='b')
plt.title('Verlauf der Area über die Anzahl der Rollouts')
plt.xlabel('Anzahl der Rollouts (N)')
plt.ylabel('Area')
plt.grid(True)
plt.tight_layout()


#----------------------------------------------
#delta Vergleich(Halbe Seitenlänge Rechteck)
#----------------------------------------------

a_max = epsilon_A_dach
b_max = epsilon_B_dach
max_abweichung = np.sqrt(a_max**2 + b_max**2)

script_ordner = os.path.dirname(os.path.abspath(__file__))


basis_ordner = "delta_ab"
unter_ordner = "rectangle_data"
voller_ordner = os.path.join(script_ordner, basis_ordner, unter_ordner)

os.makedirs(voller_ordner, exist_ok=True)


pfad = os.path.join(voller_ordner, "rectangle_abmax_data.npz")


if os.path.exists(pfad):
    try:
        
        data = np.load(pfad)["rectangle_abmax_data"]
    except (IOError, KeyError):
        print(f"Warnung: Datei '{pfad}' war fehlerhaft. Es wird ein neues leeres Datenarray erstellt.")
        data = np.empty((4, 0))
else:
    
    data = np.empty((4, 0))


neue_spalte = np.array([[a_max], [b_max], [max_abweichung], [N]])
data = np.hstack((data, neue_spalte))

sortier_indizes = np.argsort(data[1, :])
data = data[:, sortier_indizes]

np.savez(pfad, rectangle_abmax_data=data)

print("Daten erfolgreich gespeichert!")
print(f"Pfad: '{pfad}'")
print("Aktuelle Daten im Array:\n", data)