# Adapted without algorithm changes from Manas-arumalla/openarm-control
# commit 14ea59faad6706b9add5d7f8d386e43239a6360f, Apache-2.0.
import numpy as np

def quintic(t, T, q0, qd0, qf, qdf):
    """Vectorised minimum-jerk quintic with position+velocity boundary
    conditions (zero boundary acceleration). Returns (q, qd, qdd) at time ``t``.
    """
    q0 = np.asarray(q0, float); qd0 = np.asarray(qd0, float)
    qf = np.asarray(qf, float); qdf = np.asarray(qdf, float)
    if T <= 1e-6:
        return qf.copy(), qdf.copy(), np.zeros_like(qf)
    t = float(np.clip(t, 0.0, T))
    d = qf - q0
    a0, a1, a2 = q0, qd0, np.zeros_like(q0)
    a3 = (20 * d - (8 * qdf + 12 * qd0) * T) / (2 * T ** 3)
    a4 = (-30 * d + (14 * qdf + 16 * qd0) * T) / (2 * T ** 4)
    a5 = (12 * d - 6 * (qdf + qd0) * T) / (2 * T ** 5)
    q = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
    qd = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
    qdd = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3
    return q, qd, qdd
