"""Portfolio-specific pieces plugged into the generic NSGA-II engine.

Objectives (all minimized, per NSGA-II convention):
    f1 = -expected_return   (so minimizing f1 maximizes return)
    f2 = portfolio_std_dev  (risk)
    f3 = HHI = sum(w_i^2)   (concentration; minimizing it maximizes diversification)

Constraints: long-only weights summing to 1, each capped at ``MAX_WEIGHT``
to keep any single position from dominating the portfolio.
"""
import numpy as np

MAX_WEIGHT = 0.35


def project_capped_simplex(v: np.ndarray, cap: float, n_iter: int = 60) -> np.ndarray:
    """Vectorized projection of each row of v onto {w: sum(w)=1, 0<=w<=cap}."""
    v = np.clip(v, 0.0, cap)
    lo = v.min(axis=1) - cap
    hi = v.max(axis=1)
    for _ in range(n_iter):
        theta = (lo + hi) / 2.0
        w = np.clip(v - theta[:, None], 0.0, cap)
        s = w.sum(axis=1)
        too_high = s > 1.0
        lo = np.where(too_high, theta, lo)
        hi = np.where(~too_high, theta, hi)
    theta = (lo + hi) / 2.0
    w = np.clip(v - theta[:, None], 0.0, cap)
    s = w.sum(axis=1, keepdims=True)
    s[s <= 1e-12] = 1.0
    return w / s


def make_repair(cap: float = MAX_WEIGHT):
    def repair(pop: np.ndarray) -> np.ndarray:
        return project_capped_simplex(pop, cap)
    return repair


def make_evaluator(mu: np.ndarray, cov: np.ndarray):
    def evaluate(pop: np.ndarray) -> np.ndarray:
        ret = pop @ mu
        variances = np.einsum("ij,jk,ik->i", pop, cov, pop)
        variances = np.clip(variances, 0.0, None)
        risk = np.sqrt(variances)
        hhi = np.sum(pop ** 2, axis=1)
        return np.column_stack([-ret, risk, hhi])
    return evaluate


def portfolio_metrics(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> dict:
    ret = float(w @ mu)
    risk = float(np.sqrt(max(w @ cov @ w, 0.0)))
    hhi = float(np.sum(w ** 2))
    return {
        "return": ret,
        "risk": risk,
        "sharpe_like": ret / risk if risk > 1e-9 else float("nan"),
        "hhi": hhi,
        "effective_n_assets": 1.0 / hhi if hhi > 1e-9 else float("nan"),
    }
