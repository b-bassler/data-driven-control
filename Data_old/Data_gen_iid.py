import numpy as np
import os

np.random.seed(0)
    
a = 0.5
b = 0.5
x0 = 0
T = 1 #Anzahl der Datenlänge, fest bei 1 für i.i.d.
N = 10000 #Anzahl an Rollouts


mean = 0

sigma_u_quadrat = 1
sigma_u = np.sqrt(sigma_u_quadrat)

w_max = 0.01  # Maximale Abweichung der Störungen bei Uniformverteilung
sigma_w_quadrat = (w_max**2) / 3  # Varianz für die Normalverteilung, Vergleich mit gleicher Varianz
sigma_w = np.sqrt(sigma_w_quadrat)  #Standardabweichung

 
u = np.random.normal(mean, sigma_u, (N,T) ) # 2) Zufällige Eingangssignale, mean = 0, Normalverteilt mit Standardabweichung sigma_u
w = np.random.normal(mean, sigma_w, (N,T)) 
x = np.random.normal(mean, sigma_u, (N,T) )

A = np.array([[a]])  
B = np.array([[b]])


y = np.zeros((N, T)) # leerer Datenvektor y

for k in range(N): 
    y[k,:] = a * x[k, :] + b * u[k,:] + w[k,:]

output_folder_name = "generated_data"


script_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(script_dir, output_folder_name)


os.makedirs(output_dir, exist_ok=True)
print(f"Speichere Daten im Ordner: {output_dir}")


np.save(os.path.join(output_dir, "sigma_u_iid.npy"), sigma_u)
np.save(os.path.join(output_dir, "sigma_w_iid.npy"), sigma_w)
np.save(os.path.join(output_dir, "sigma_w_quadrat_iid.npy"), sigma_w_quadrat)
np.save(os.path.join(output_dir, "iid_state_data_gauss.npy"), x)
np.save(os.path.join(output_dir, "iid_input_data_gauss.npy"), u)
np.save(os.path.join(output_dir, "iid_noise_data_gauss.npy"), w)
np.save(os.path.join(output_dir, "iid_output_data_gauss.npy"), y)

np.save(os.path.join(output_dir, "A_Matrix.npy"), A)
np.save(os.path.join(output_dir, "B_Matrix.npy"), B)


print("Daten erfolgreich gespeichert. 👍")
