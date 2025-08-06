import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. Daten laden ---
# Annahme: Die Daten wurden zuvor generiert und gespeichert.
# Ihr Code zum Laden der Daten ist hier korrekt.
try:
    data_folder = r"C:\Users\benno\Desktop\Simulations\Data\generated_data"
    if not os.path.exists(data_folder):
        print(f"Warnung: Der Datenordner '{data_folder}' existiert nicht.")
    else:
        daten_dict = {
            f: np.load(os.path.join(data_folder, f))
            for f in os.listdir(data_folder)
            if f.endswith(".npy")
        }
        state_data = daten_dict["iid_state_data_gauss.npy"]
        input_data = daten_dict["iid_input_data_gauss.npy"]
        noise_data = daten_dict["iid_noise_data_gauss.npy"]
        output_data = daten_dict["iid_output_data_gauss.npy"]
        A_true = daten_dict["A_Matrix.npy"]
        B_true = daten_dict["B_Matrix.npy"]
        sigma_u = daten_dict["sigma_u_iid.npy"]
        sigma_w = daten_dict["sigma_w_iid.npy"]
except (FileNotFoundError, KeyError) as e:
    print(f"Fehler beim Laden der Daten: {e}")
    print("Stellen Sie sicher, dass alle .npy-Dateien im angegebenen Ordner vorhanden sind.")
    exit()


# --- 2. Daten für die Analyse vorbereiten ---
T = 100  # Anzahl an verwendeten Datenpunkten

# Wir nehmen die ersten T Datenpunkte aus den geladenen Arrays
x = state_data[:T, :]
u = input_data[:T, :]
y = output_data[:T, :]

# --- 3. Funktionen für Schätzung und Visualisierung ---

def least_squares_estimator_iid(x_data, u_data, y_data):
    """
    Führt die Least-Squares-Schätzung für i.i.d. Daten durch.

    Args:
        x_data (np.array): Array der Zustandsdaten mit Shape (T, n).
        u_data (np.array): Array der Eingabedaten mit Shape (T, p).
        y_data (np.array): Array der Ausgabedaten mit Shape (T, n).

    Returns:
        tuple: Ein Tupel mit den geschätzten Matrizen (A_est, B_est).
    """
    # Die Regressormatrix Z hat die Form (T, n+p)
    # Jede Zeile ist [x_i, u_i]
    Z = np.hstack([x_data, u_data])

    # Schätzung durchführen (pinv ist robuster als inv)
    # Formel: theta = (Z^T * Z)^-1 * Z^T * y
    try:
        theta_est = np.linalg.pinv(Z.T @ Z) @ Z.T @ y_data
    except np.linalg.LinAlgError:
        print("Fehler: Die Matrix Z.T @ Z ist singulär.")
        return None, None

    # Dimensionen aus den Daten ableiten
    n = x_data.shape[1]
    p = u_data.shape[1]

    # Geschätzte Parameter extrahieren
    A_est = theta_est[:n, :]
    B_est = theta_est[n:, :]

    return A_est, B_est

def plot_confidence_ellipse(a_true, b_true, a_hat, b_hat, P_matrix, delta):
    """
    Visualisiert die Konfidenz-Ellipse.

    Args:
        a_true, b_true (float): Wahre Parameter.
        a_hat, b_hat (float): Geschätzte Parameter (Zentrum der Ellipse).
        P_matrix (np.array): Die 2x2 Formmatrix der Ellipse.
        delta (float): Das Konfidenzniveau (z.B. 0.05 für 95%).
    """
    # Eigendekomposition der Formmatrix P
    eigenvalues, eigenvectors = np.linalg.eig(P_matrix)
    
    # Halbachsenlängen der Ellipse
    semi_axis_1 = 1 / np.sqrt(eigenvalues[0])
    semi_axis_2 = 1 / np.sqrt(eigenvalues[1])

    # Rotationswinkel der Ellipse
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Punkte auf einem Einheitskreis generieren
    phi = np.linspace(0, 2 * np.pi, 100)
    circle_points = np.vstack([np.cos(phi), np.sin(phi)])

    # Transformation zur finalen Ellipse (Skalieren, Rotieren, Verschieben)
    ellipse_transform = eigenvectors @ np.diag([semi_axis_1, semi_axis_2])
    ellipse_points = ellipse_transform @ circle_points
    ellipse_a = ellipse_points[0, :] + a_hat
    ellipse_b = ellipse_points[1, :] + b_hat

    # Plot erstellen
    plt.figure(figsize=(10, 8))
    plt.plot(ellipse_a, ellipse_b, label=f'{100*(1-delta):.0f}% Konfidenz-Ellipse', color='blue')
    plt.fill(ellipse_a, ellipse_b, alpha=0.2, color='blue')
    plt.scatter(a_true, b_true, color='red', marker='x', s=120, zorder=5, label='Wahre Parameter (a, b)')
    plt.scatter(a_hat, b_hat, color='green', marker='+', s=120, zorder=5, label='Geschätzte Parameter (â, b̂)')

    plt.title(f'Konfidenz-Ellipse aus Proposition 2.4 (T = {T})')
    plt.xlabel('Parameter a')
    plt.ylabel('Parameter b')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# --- 4. Hauptskript ausführen ---

# Schätzer berechnen
# Da A und B Skalare sind, müssen wir sie für die Funktion in 2D-Arrays umwandeln
A_est_mat, B_est_mat = least_squares_estimator_iid(x, u, y)

if A_est_mat is not None:
    # Ergebnisse extrahieren
    a_hat = A_est_mat[0, 0]
    b_hat = B_est_mat[0, 0]
    print(f"Geschätzte Parameter: a_hat = {a_hat:.4f}, b_hat = {b_hat:.4f}")
    print(f"Wahre Parameter:      a_true = {A_true[0,0]:.4f}, b_true = {B_true[0,0]:.4f}")

    # Parameter für die Ellipse berechnen
    n_dim = x.shape[1]
    p_dim = u.shape[1]
    delta = 0.05  # 95% Konfidenz

    # Konstante C aus der Proposition
    C_const = sigma_w**2 * (np.sqrt(n_dim + p_dim) + np.sqrt(n_dim) + np.sqrt(2 * np.log(1/delta)))**2

    # Gram'sche Matrix Z^T * Z
    Z = np.hstack([x, u])
    gram_matrix = Z.T @ Z

    # Formmatrix P der Ellipse
    P_ellipse = gram_matrix / C_const

    # Visualisierung
    plot_confidence_ellipse(A_true[0,0], B_true[0,0], a_hat, b_hat, P_ellipse, delta)





#----------------------------------------------
# Bestimmung von max_a und max_b
#----------------------------------------------

# 1. Berechne die Ellipsen-Geometrie aus der Formmatrix P
eigenvalues, eigenvectors = np.linalg.eig(P_ellipse)

# Die Halbachsenlängen sind die Wurzel aus dem Kehrwert der Eigenwerte
semi_axis_1 = np.sqrt(1 / eigenvalues[0])
semi_axis_2 = np.sqrt(1 / eigenvalues[1])

# 2. Generiere die Punkte der Ellipse relativ zum Zentrum (0,0)
# Wir brauchen genügend Punkte für eine gute Genauigkeit
phi = np.linspace(0, 2 * np.pi, 1000) 
circle_points = np.vstack([np.cos(phi), np.sin(phi)])

# Transformationsmatrix anwenden (Rotation und Skalierung)
ellipse_transform = eigenvectors @ np.diag([semi_axis_1, semi_axis_2])
ellipse_points_centered = ellipse_transform @ circle_points

# 3. Finde die maximale absolute Koordinate entlang jeder Achse
# Das entspricht der halben Breite/Höhe der Bounding Box der Ellipse.
max_a = np.max(np.abs(ellipse_points_centered[0, :]))
max_b = np.max(np.abs(ellipse_points_centered[1, :]))




# Finde den Index des KLEINSTEN Eigenwerts von P_ellipse
min_eigenvalue_index = np.argmin(eigenvalues)

# Die längste Halbachse ist 1 / sqrt(kleinster Eigenwert)
worst_case_deviation = np.sqrt(1 / eigenvalues[min_eigenvalue_index])

print(f"Worst-Case Abweichung (lange Halbachse): {worst_case_deviation:.4f}")
print(f"Eigenwerte: {eigenvalues}")
print(f"Index des kleinsten Eigenwerts: {min_eigenvalue_index}")
print(f"Längste Halbachse (Worst Case): {worst_case_deviation:.4f}")

#----------------------------------------------
#delta Vergleich (Maximale Abweichung der Ellipse in a und b Richtung)
#----------------------------------------------



# --- 5. SPEICHERN DER ELLIPSEN-EIGENSCHAFTEN ---

print("\n--- Speichere Ellipsen-Eigenschaften ---")

# 1. Benötigte Variablen aus den vorherigen Berechnungen zuweisen

# Zentrum der Ellipse
zentrum = [a_hat, b_hat]

# Halbachsenlängen (bereits berechnet als semi_axis_1 und semi_axis_2)
# Wir sortieren sie, um eine konsistente Reihenfolge zu haben (größte zuerst)
halbachsen = sorted([semi_axis_1, semi_axis_2], reverse=True)

# Rotationswinkel aus den Eigenvektoren berechnen und in Grad umwandeln
# eigenvectors wurde bereits aus P_ellipse berechnet
winkel_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
winkel_grad = np.degrees(winkel_rad)

# Ausgabe der berechneten Werte zur Kontrolle
print(f"Zentrum (a_hat, b_hat): ({zentrum[0]:.5f}, {zentrum[1]:.5f})")
print(f"Länge der Halbachsen: {halbachsen[0]:.5f} und {halbachsen[1]:.5f}")
print(f"Rotation der Hauptachse: {winkel_grad:.5f}°")
print("-------------------------------------------\n")


# 2. Speicherlogik anwenden (exakt wie im vorherigen Skript)

# Name des Ordners, in dem die Ergebnisse gespeichert werden
ordner_name = "results"

# Erstelle den Ordner, falls er noch nicht existiert
os.makedirs(ordner_name, exist_ok=True)

# Dateiname für die Simulationsergebnisse
dateiname = "simulation_ergebnis_2.npz"
voller_pfad = os.path.join(ordner_name, dateiname)

# Speichere die charakteristischen Daten der Ellipse in einer .npz-Datei
np.savez(voller_pfad,
         zentrum=zentrum,
         halbachsen=halbachsen,
         winkel_grad=winkel_grad)

print(f"✅ Ergebnisse erfolgreich in '{voller_pfad}' gespeichert.")























script_ordner = os.path.dirname(os.path.abspath(__file__))


basis_ordner = "delta_ab"
unter_ordner = "ellipse_data"
voller_ordner = os.path.join(script_ordner, basis_ordner, unter_ordner)

os.makedirs(voller_ordner, exist_ok=True) 


pfad = os.path.join(voller_ordner, "ellipse_dean_abmax_data.npz")


if os.path.exists(pfad):
    try:
        
        data = np.load(pfad)["ellipse_dean_abmax_data"]
    except (IOError, KeyError):
        print(f"Warnung: Datei '{pfad}' war fehlerhaft. Es wird ein neues leeres Datenarray erstellt.")
        data = np.empty((4, 0))
else:
    
    data = np.empty((4, 0))


neue_spalte = np.array([[max_a], [max_b], [worst_case_deviation], [T]])
data = np.hstack((data, neue_spalte))

sortier_indizes = np.argsort(data[1, :])
data = data[:, sortier_indizes]

np.savez(pfad, ellipse_dean_abmax_data=data)

print("Daten erfolgreich gespeichert!")
print(f"Pfad: '{pfad}'")
print("Aktuelle Daten im Array:\n", data)


