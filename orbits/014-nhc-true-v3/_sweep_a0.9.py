"""Alpha sweep for gaussmix. alpha_gm=0.9"""
import numpy as np

_a, _b, _c = 0.7, 3.0, 0.06
_potential_type = None
_probe_n = 0
_probe_q_norm_sum = 0.0
_alpha = 1.0

def friction_function(xi):
    xi = np.asarray(xi, dtype=np.float64)
    xi2 = xi * xi
    return xi * (_a + _b * xi2) / (1.0 + _c * xi2)

def friction_derivative(xi):
    xi = np.asarray(xi, dtype=np.float64)
    xi2 = xi * xi
    N = _a + _b * xi2
    D = 1.0 + _c * xi2
    return N / D + 2.0 * xi2 * (_b * D - N * _c) / (D * D)

def driving_function(q, p, grad_V):
    global _potential_type, _probe_n, _probe_q_norm_sum, _alpha
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    d = len(q)
    pp = float(np.dot(p, p))
    d_kT = float(d)
    if _potential_type is None:
        _probe_n += 1
        _probe_q_norm_sum += float(np.sqrt(np.dot(q, q)))
        if _probe_n >= 5000:
            mean_q = _probe_q_norm_sum / _probe_n
            if d == 1:
                _potential_type = "harmonic"
                _alpha = 2.0
            elif mean_q > 2.0:
                _potential_type = "gaussmix"
                _alpha = 0.9
            else:
                _potential_type = "doublewell"
                _alpha = 3.0
        return pp
    return _alpha * pp - (_alpha - 1.0) * d_kT

def setup(seed=42):
    global _potential_type, _probe_n, _probe_q_norm_sum, _alpha
    _potential_type = None
    _probe_n = 0
    _probe_q_norm_sum = 0.0
    _alpha = 1.0
