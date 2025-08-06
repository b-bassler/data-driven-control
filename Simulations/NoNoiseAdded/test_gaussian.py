import numpy as np
import cvxpy as cp
import matplotlib as plt
import matplotlib.pyplot as plt

#Number of datasamples used (T+1)
T = 50

X = np.load("state_data.npy")

X_plus = X[:, 1:T+1]
X_minus = X[:, 0:T]
U_minus = np.load("input_data.npy")[:, 0:T]
print("Shape of X_plus:", X_plus.shape)
print("Shape of X_minus:", X_minus.shape)
print("Shape of U_minus:", U_minus.shape)


A_min = 0.2
A_max = 0.4
B_min = 0.7
B_max = 0.9

sample_size = 800

#Nosemodel, Random uniform noise with maximum amplitude w_max
w_max = 0.01

Qw = -np.eye(X.shape[0]) 
Sw = np.zeros((X.shape[0], X_minus.shape[1]))
Rw = w_max**2 * np.eye(X_minus.shape[1]) 

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


if valid_pairs:
    A_ok, B_ok = zip(*valid_pairs)
    plt.figure()
    plt.scatter(0.3, 0.8, c="red", marker="x", s=100, label="(A=0.3, B=0.8)", zorder=2)
    plt.scatter(A_ok, B_ok, s=15)
    plt.xlim(0.27, 0.33)   
    plt.ylim(0.77, 0.83)
    plt.xlabel("A")
    plt.ylabel("B")
    plt.title("Matching (A,B)")
    plt.grid(True)
    plt.show()
    plt.scatter(A_ok, B_ok, s=15, label="True values")
   
else:
    print("Keine (A,B)-Paare mit allen Eigenwerten > 0 gefunden.")