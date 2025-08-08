import numpy as np
import os
np.random.seed(0)

a = 0.5
b = 0.5
x0 = 0
T = 10000 #Anzahl der Zeitschritte
N = 1 #Anzahl an Rollouts


mean = 0

sigma_u_quadrat = 1
sigma_u = np.sqrt(sigma_u_quadrat)

w_max = 0.01  # Maximale Abweichung der Störungen bei Uniformverteilung
sigma_w_quadrat = (w_max**2) / 3  # Varianz für die Normalverteilung
sigma_w = np.sqrt(sigma_w_quadrat)  #Standardabweichung

 
u = np.random.normal(mean, sigma_u, (N,T) ) # 2) Zufällige Eingangssignale, mean = 0, Normalverteilt mit Standardabweichung sigma_u
w = np.random.normal(mean, sigma_w, (N,T)) 


print("sigma_w:", sigma_w)
print("sigma_u:", sigma_u)


A = np.array([[a]])  
B = np.array([[b]])



x = np.zeros((N, T+1))    #leeren Zustandsvektor x erzeugen, mit Anfangsbedingung x0
x[:, 0] = x0

for i in range(N):
    for k in range(T):                          
        x[i, k+1] = a * x[i, k] + b * u[i, k] + w[i, k]


print("Shape of x:", x.shape)

output_folder_name = "generated_data"


script_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(script_dir, output_folder_name)


os.makedirs(output_dir, exist_ok=True)
print(f"Speichere Daten im Ordner: {output_dir}")


np.save(os.path.join(output_dir, "sigma_u2"), sigma_u)
np.save(os.path.join(output_dir, "sigma_w2"), sigma_w)
np.save(os.path.join(output_dir, "sigma_w_quadrat2"), sigma_w_quadrat)
np.save(os.path.join(output_dir, "state_data_gauss2.npy"), x)
np.save(os.path.join(output_dir, "input_data_gauss2.npy"), u)
np.save(os.path.join(output_dir, "noise_data_gauss2.npy"), w)
np.save(os.path.join(output_dir, "A_Matrix.npy"), A)
np.save(os.path.join(output_dir, "B_Matrix.npy"), B)


print("Daten erfolgreich gespeichert. 👍")








