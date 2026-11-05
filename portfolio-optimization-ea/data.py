"""Synthetic market data generator.

Builds a realistic daily-return panel for a universe of assets using a
one-factor (market) + sector-factor model, then derives the annualized
expected-return vector and covariance matrix used by the optimizer.
"""
import numpy as np
import pandas as pd

SECTORS = {
    "Tech": ["TCH1", "TCH2", "TCH3"],
    "Finance": ["FIN1", "FIN2", "FIN3"],
    "Healthcare": ["HLT1", "HLT2"],
    "Energy": ["ENG1", "ENG2"],
    "Consumer": ["CON1", "CON2"],
    "Utilities": ["UTL1", "UTL2"],
}

ASSETS = [t for tickers in SECTORS.values() for t in tickers]
N_ASSETS = len(ASSETS)


def generate_returns(n_days: int = 756, seed: int = 42) -> pd.DataFrame:
    """Simulate daily returns via market + sector + idiosyncratic factors."""
    rng = np.random.default_rng(seed)

    market = rng.normal(0.0004, 0.010, n_days)
    sector_factors = {
        sector: rng.normal(0.0001, 0.006, n_days) for sector in SECTORS
    }

    # Per-asset base annual drift and loadings, varied so the Pareto front
    # has genuine high-return/high-risk vs. low-return/low-risk trade-offs.
    base_drift = rng.uniform(0.00015, 0.00075, N_ASSETS)
    market_beta = rng.uniform(0.6, 1.6, N_ASSETS)
    sector_beta = rng.uniform(0.5, 1.3, N_ASSETS)
    idio_vol = rng.uniform(0.006, 0.020, N_ASSETS)

    data = {}
    idx = 0
    for sector, tickers in SECTORS.items():
        for ticker in tickers:
            idio = rng.normal(0.0, idio_vol[idx], n_days)
            r = (
                base_drift[idx]
                + market_beta[idx] * market
                + sector_beta[idx] * sector_factors[sector]
                + idio
            )
            data[ticker] = r
            idx += 1

    dates = pd.bdate_range("2023-01-02", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=ASSETS)


def annualized_stats(returns: pd.DataFrame):
    mu = returns.mean().to_numpy() * 252
    cov = returns.cov().to_numpy() * 252
    return mu, cov


if __name__ == "__main__":
    df = generate_returns()
    df.to_csv("outputs/daily_returns.csv")
    mu, cov = annualized_stats(df)
    pd.Series(mu, index=ASSETS, name="expected_annual_return").to_csv(
        "outputs/expected_returns.csv"
    )
    pd.DataFrame(cov, index=ASSETS, columns=ASSETS).to_csv(
        "outputs/covariance_matrix.csv"
    )
    print(f"Generated {len(df)} trading days for {N_ASSETS} assets.")
    print("Expected annual returns:")
    print(pd.Series(mu, index=ASSETS).round(4))
