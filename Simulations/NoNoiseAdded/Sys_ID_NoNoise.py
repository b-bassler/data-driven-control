import numpy as np
import cvxpy as cp


X = np.load("state_data.npy")

X_plus = X[:, 1:20]
X_minus = X[:, 0:19]
U_minus = np.load("input_data.npy")[:, 0:19]

#Check if Data is persistently exciting
Rank_Matrix = np.block([[X_minus],[U_minus]])

if np.linalg.matrix_rank(Rank_Matrix) == (X_minus.shape[0] + U_minus.shape[0]):
    print("Data is persistently exciting")
else:
    print("Data is not persistently exciting")

#print (Rank_Matrix)
#print(np.linalg.matrix_rank(Rank_Matrix))


AB = X_plus @ np.linalg.pinv(Rank_Matrix)
A = AB[:, :X_minus.shape[0]]
B = AB[:, X_minus.shape[0]:]
print("A:\n", A)
print("B:\n", B)


