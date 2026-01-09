import numpy as np

class UKF:
    def __init__(self, dim_x, dim_z, f, h, Q, R, dt):
        self.n = dim_x
        self.m = dim_z
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.dt = dt

        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)

        # UKF parameters
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n

        self.gamma = np.sqrt(self.n + self.lambda_)

        self.Wm = np.full(2 * self.n + 1, 1. / (2 * (self.n + self.lambda_)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

    def sigma_points(self, x, P):
        # Robust Cholesky: add tiny jitter until factorization succeeds, fallback to sqrt-diagonal
        eps = 1e-12
        for _ in range(6):
            try:
                U = np.linalg.cholesky(P + eps * np.eye(self.n))
                break
            except np.linalg.LinAlgError:
                eps *= 100
        else:
            # Fallback: use sqrt of diagonal (keeps shape but loses cross-covariance)
            U = np.diag(np.sqrt(np.maximum(np.diag(P), 1e-8)))

        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = x
        for i in range(self.n):
            sigmas[i + 1]     = x + self.gamma * U[:, i]
            sigmas[i + 1 + self.n] = x - self.gamma * U[:, i]
        return sigmas

    def predict(self):
        sigmas = self.sigma_points(self.x, self.P)
        sigmas_f = np.array([self.f(s, self.dt) for s in sigmas])

        self.x = np.sum(self.Wm[:, None] * sigmas_f, axis=0)

        self.P = self.Q.copy()
        for i in range(sigmas_f.shape[0]):
            dx = sigmas_f[i] - self.x
            self.P += self.Wc[i] * np.outer(dx, dx)

        self.sigmas_f = sigmas_f

    def update(self, z):
        Zsig = np.array([self.h(s) for s in self.sigmas_f])
        z_pred = np.sum(self.Wm[:, None] * Zsig, axis=0)

        S = self.R.copy()
        for i in range(Zsig.shape[0]):
            dz = Zsig[i] - z_pred
            dz[1] = np.arctan2(np.sin(dz[1]), np.cos(dz[1]))
            S += self.Wc[i] * np.outer(dz, dz)

        Pxz = np.zeros((self.n, self.m))
        for i in range(Zsig.shape[0]):
            dx = self.sigmas_f[i] - self.x
            dz = Zsig[i] - z_pred
            dz[1] = np.arctan2(np.sin(dz[1]), np.cos(dz[1]))
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # Use pseudo-inverse for numerical robustness
        K = Pxz @ np.linalg.pinv(S)

        dz = z - z_pred
        dz[1] = np.arctan2(np.sin(dz[1]), np.cos(dz[1]))

        self.x = self.x + K @ dz
        self.P = self.P - K @ S @ K.T