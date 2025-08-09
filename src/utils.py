import numpy as np

def calculate_norm_error(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """Calculates the L2 norm of the difference between two matrices."""
    error = np.linalg.norm(matrix1 - matrix2, ord=2)
    return float(error)

