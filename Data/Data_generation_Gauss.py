import numpy as np
import os
np.random.seed(1)


a = 0.3
b = 0.8
x0 = 0
T = 100000 

mean = 0
w_max = 0.01  # Maximale Abweichung der Störungen bei Uniformverteilung
sigma_w_quadrat = (w_max**2) / 3  # Varianz für die Normalverteilung

z = 1.96 #Z-Quantil für 1-delta = 0.95
a = w_max
sigma_w = a / z

sigma_quadrat = sigma_w**2


#sigma_w = np.sqrt(sigma_w_quadrat)

abweichung = 1
 
u = np.random.normal(mean ,abweichung, (1,T))  # 2) Zufällige Eingangssignale zwischen 0 und 1
sigma_u_quadrat = (2*abweichung)**2 / 12
sigma_u = np.sqrt(sigma_u_quadrat)
w = np.random.normal(mean, sigma_w, (1, T)) 


print("sigma_w:", sigma_w)
print("sigma_u:", sigma_u)


A = np.array([[a]])  
B = np.array([[b]])



x = np.zeros((1, T+1))    #leeren Zustandsvektor x erzeugen, mit Anfangsbedingung x0
x[:, 0] = x0

for k in range(T):                          # x(k+1) = A*x(k) + B*u(k) + w(k)
    x[:, k+1] = A @ x[:, k] + B @ u[:, k] + w[:, k]

print("Zustandsvektor x:\n", x)
print(x.shape)


output_folder_name = "generated_data"

script_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(script_dir, output_folder_name)


os.makedirs(output_dir, exist_ok=True)
print(f"Speichere Daten im Ordner: {output_dir}")


np.save(os.path.join(output_dir, "sigma_u"), sigma_u)
np.save(os.path.join(output_dir, "sigma_w"), sigma_w)
np.save(os.path.join(output_dir, "sigma_w_quadrat"), sigma_w_quadrat)
np.save(os.path.join(output_dir, "state_data_gauss.npy"), x)
np.save(os.path.join(output_dir, "input_data_gauss.npy"), u)
np.save(os.path.join(output_dir, "noise_data_gauss.npy"), w)


print("Daten erfolgreich gespeichert. 👍")