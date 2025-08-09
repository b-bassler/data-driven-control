import numpy as np
import matplotlib.pyplot as plt
from rsome import ro
from rsome import cpt_solver as cpt
import rsome as rso

model = ro.Model()

P = model.dvar((2, 2))
c = model.dvar(2)
Z = rso.tril(model.dvar((2, 2)))
v = model.dvar(2)

model.max(v.sum())
model.st(v <= rso.log(rso.diag(Z)))
model.st(rso.rstack([P, Z], 
                    [Z.T, rso.diag(Z, fill=True)]) >> 0)
for i in range(m):
    model.st(rso.norm(P@xs[i] - c) <= 1)
model.st(P >> 0)

model.solve(cpt)
print(f'Determinant: {np.exp(model.get())}')

Ps = P.get()
cs = c.get()

step = 0.01
t = np.arange(0, 2*np.pi+step, step)
y = np.vstack((np.cos(t), np.sin(t))).T

ellip = np.linalg.inv(Ps) @ (y + cs).T

plt.figure(figsize=(5, 5))
plt.scatter(xs[:, 0], xs[:, 1], 
            marker='o', facecolor='none', color='k', label='Data points')
plt.plot(ellip[0], ellip[1], color='b', 
         label='Minimum enclosing ellipsoid')
plt.legend(fontsize=12, bbox_to_anchor=(1.01, 1.02))
plt.axis('equal')
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.show()