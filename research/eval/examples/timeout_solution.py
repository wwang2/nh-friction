"""Timeout test — setup() sleeps forever."""
import numpy as np, time

def setup(seed: int) -> None:
    time.sleep(9999)

def friction_function(xi: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(xi, dtype=float))

def friction_derivative(xi: np.ndarray) -> np.ndarray:
    x = np.asarray(xi, dtype=float)
    return 1.0 - np.tanh(x)**2
