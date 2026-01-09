import numpy as np
from scipy.optimize import minimize_scalar

def covariance_intersection(x1, P1, x2, P2):
    # Robust fusion using Covariance Intersection (CI)
    # Optimizes omega to minimize the trace of the resulting covariance
    def objective(omega):
        P_inv = omega * np.linalg.pinv(P1) + (1 - omega) * np.linalg.pinv(P2)
        P_f = np.linalg.pinv(P_inv)
        return np.trace(P_f)

    res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    w = res.x
    
    invP1 = np.linalg.pinv(P1)
    invP2 = np.linalg.pinv(P2)
    
    P_f_inv = w * invP1 + (1 - w) * invP2
    P_f = np.linalg.pinv(P_f_inv)
    x_f = P_f @ (w * invP1 @ x1 + (1 - w) * invP2 @ x2)
    
    return x_f, P_f, w