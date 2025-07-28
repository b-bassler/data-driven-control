import numpy as np


def min_enclosing_ellipse(points: np.ndarray, tol: float = 1e-3) -> tuple[np.ndarray, float, float, float]:
    """
    Berechnet Zentrum, Halbachsen (a,b) und Rotationswinkel phi der minimalen Ellipse,
    die alle Punkte umschließt (Khachiyan-Algorithmus).

    Args:
        points: (N,2)-Array von Punkten
        tol:  Abbruchschwelle für die Iteration

    Returns:
        center: (2,) – Ellipsenzentrum
        a, b: Halbachsenlängen
        phi:  Rotationswinkel (in rad)
    """     
    P = points.T               # Form: 2×N
    N = P.shape[1]
    d = 2
    Q = np.vstack((P, np.ones((1, N))))      # Form: 3×N
    u = np.ones(N) / N
    err = tol + 1

    # Khachiyan-Iteration
    while err > tol:
        X = Q @ np.diag(u) @ Q.T
        M = np.diag(Q.T @ np.linalg.inv(X) @ Q)
        j = np.argmax(M)
        step = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))
        u_new = (1 - step) * u
        u_new[j] += step
        err = np.linalg.norm(u_new - u)
        u = u_new

    # Zentrum berechnen
    center = P @ u

    # Matrix A der Ellipsengleichung (x−c)ᵀ A (x−c) = 1
    A = np.linalg.inv(P @ np.diag(u) @ P.T - np.outer(center, center)) / d

    # Eigenzerlegung für Achsen und Winkel
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    a, b = 1 / np.sqrt(eigvals)             # Halbachsen
    phi = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    return center, a, b, phi


if __name__ == "__main__":
    # Beispiel: mindestens 3 nicht-kolineare Punkte
    pts = np.array([
        (0.27646076794657765, 0.8308848080133556),
        (0.5, 0.2),
        (0.1, 0.7),
    ], dtype=np.float64)
    c, a, b, phi = min_enclosing_ellipse(pts)
    print(f"Zentrum: {c}, a={a:.3f}, b={b:.3f}, phi={phi:.3f} rad")
