import numpy as np
import os   
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


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

A_true = daten_dict["A_Matrix.npy"]
B_true = daten_dict["B_Matrix.npy"]




N = 50 #Anzahl Rollouts
T = 50 #Länge des Rollouts
#X = state_data[0:1, 1:T+1]


#X_vertical = X.T
#X_minus = state_data[0:1, 0:T]
#U_minus = input_data[0:1, 0:T]

X_ges_plus = state_data[:N, 1:T+1]  # von x1 bis xT
X_ges_minus = state_data[:N, 0:T]   # von x0 bis xT-1
U_ges_minus = input_data[:N, 0:T]   # von x0 bis xT-1


X_N = X_ges_plus.reshape(-1, 1)
X_N_minus = X_ges_minus.reshape(-1, 1)
U_N_minus = U_ges_minus.reshape(-1, 1)








#Z = np.block([[X_minus.T, U_minus.T]]) 
Z = np.block([[X_N_minus, U_N_minus]])               
 

AB_est = np.linalg.inv(Z.T @ Z) @ Z.T @ X_N

print(AB_est)
A_est = AB_est[:1, :]
B_est = AB_est[1:2, :]
print("Estimated A matrix:\n", A_est)
print("Estimated B matrix:\n", B_est)




G_T = np.hstack([
    np.linalg.matrix_power(A_true, k) @ B_true
    for k in range(T, -1, -1)
])
F_T = np.hstack([
    np.linalg.matrix_power(A_est, k) for k in range(T, -1, -1)])


delta = 0.1
n = 1
p = 1



e_A = (
    (16 * sigma_w)
    / (np.sqrt(
        np.min(np.linalg.eigvals(sigma_u**2 * (G_T @ G_T.T)
            + sigma_w**2 * (F_T @ F_T.T))))
    )
) * np.sqrt((n + 2*p) * np.log(36/delta) / N)

print("e_A:", e_A)


e_B = (
    (16 * sigma_w) / (sigma_u)
    * np.sqrt(((n + 2 * p) * np.log(36 / delta)) / N)
)

print("e_B:", e_B)


e_A = e_A.item()
e_B = e_B.item()




fig, ax = plt.subplots()

# Rechteck-Koordinaten: untere linke Ecke und Breite/Höhe
rect = Rectangle(
    (A_est - e_A, B_est - e_B),  # (x0, y0)
    2*e_A,                       # Breite
    2*e_B,                       # Höhe
    edgecolor='black',
    facecolor='blue',
    alpha = 0.5
)
ax.add_patch(rect)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.plot(
    A_est,
    B_est,
    marker='x',    
    color='red',   
    linestyle='None',  
    markersize=5  
)
ax.set_xlim(0.1, 0.5)  # xmin, xmax
ax.set_ylim(0.6, 1)  # ymin, ymax
ax.set_xlabel('A_est')
ax.set_ylabel('B_est')
ax.set_aspect('equal', 'box')
plt.show()