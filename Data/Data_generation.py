import numpy as np
import os

np.random.seed(0)

a = 0.3
b = 0.8
x0 = np.random.rand(1)
T = 1000 #Anzahl der Zeitschritte
u = np.random.rand(1, T)  # 2) Zufällige Eingangssignale zwischen 0 und 1

A = np.array([[a]])  
B = np.array([[b]])



x = np.zeros((1, T+1))    #leeren Zustandsvektor x erzeugen, mit Anfangsbedingung x0
x[:, 0] = x0

for k in range(T):                          # x(k+1) = A*x(k) + B*u(k) + w(k)
    x[:, k+1] = A @ x[:, k] + B @ u[:, k] 
print("Zustandsvektor x:\n", x)
print(x.shape)

# 1. Namen für den Unterordner festlegen 📁
output_folder_name = "generated_data"

# 2. Den Pfad zum aktuellen Skript-Verzeichnis ermitteln
script_dir = os.path.dirname(os.path.abspath(__file__))

# 3. Den vollständigen Pfad zum neuen Unterordner erstellen
output_dir = os.path.join(script_dir, output_folder_name)

# 4. Sicherstellen, dass der Unterordner existiert (erstellt ihn, falls nicht)
os.makedirs(output_dir, exist_ok=True)
print(f"Speichere Daten im Ordner: {output_dir}")

# 5. Arrays im neuen Unterordner speichern
np.save(os.path.join(output_dir, "state_data.npy"), x)
np.save(os.path.join(output_dir, "input_data.npy"), u)

print("Daten erfolgreich gespeichert. 👍")