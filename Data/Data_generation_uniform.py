import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(1)

a = 0.5
b = 0.5
x0 = np.random.rand(1)
T = 100000 #Anzahl der Zeitschritte
u = np.random.rand(1, T)  # 2) Zufällige Eingangssignale zwischen 0 und 1


A = np.array([[a]])  
B = np.array([[b]])
a = 0.01 #Intervallgröße
w = np.random.uniform(-a, a, (1, T))  # 1) Zufällige Störungen im Bereich [-0.01, 0.01]


x = np.zeros((1, T+1))    #leeren Zustandsvektor x erzeugen, mit Anfangsbedingung x0
x[:, 0] = x0

for k in range(T):                          # x(k+1) = A*x(k) + B*u(k) + w(k)
    x[:, k+1] = A @ x[:, k] + B @ u[:, k] + w[:, k]



output_folder_name = "generated_data"


script_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(script_dir, output_folder_name)


os.makedirs(output_dir, exist_ok=True)
print(f"Speichere Daten im Ordner: {output_dir}")


np.save(os.path.join(output_dir, "state_data_uniform.npy"), x)
np.save(os.path.join(output_dir, "input_data_uniform.npy"), u)
np.save(os.path.join(output_dir, "noise_data_uniform.npy"), w)

print("Daten erfolgreich gespeichert. 👍")

plt.plot(x[0, :1000])
plt.title("Trajektorie der ersten 1000 Einträge von x")
plt.xlabel("Zeitschritt (Index 0-499)")
plt.ylabel("Wert von x")
plt.grid(True)

plt.show()  