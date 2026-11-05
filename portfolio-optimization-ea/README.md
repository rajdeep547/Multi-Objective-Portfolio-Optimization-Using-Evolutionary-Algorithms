# Multi-Objective Portfolio Optimization Using Evolutionary Algorithms

Builds an investment portfolio that balances **expected return**, **risk**,
and **diversification** using a from-scratch NSGA-II (Non-dominated Sorting
Genetic Algorithm II) implementation.

## Problem formulation

Given a universe of 14 synthetic assets across 6 sectors, find portfolio
weight vectors `w` (long-only, `sum(w) = 1`, each position capped at 35%)
that minimize:

- `f1 = -expected_return`  (maximize return)
- `f2 = portfolio_std_dev` (minimize risk/volatility)
- `f3 = HHI = sum(w_i^2)`  (minimize concentration -> maximize diversification)

NSGA-II evolves a population of candidate portfolios using simulated binary
crossover, polynomial mutation, non-dominated sorting, and crowding-distance
selection, converging to the Pareto front: the set of portfolios where no
objective can be improved without worsening another.

## Files

- `data.py` — simulates a daily-return panel via a market + sector factor
  model and derives annualized expected returns / covariance.
- `nsga2.py` — generic, reusable NSGA-II engine (sorting, crowding distance,
  SBX crossover, polynomial mutation).
- `portfolio.py` — the portfolio-specific objective function and the
  capped-simplex repair operator that keeps every candidate a valid
  long-only portfolio summing to 1.
- `main.py` — runs the optimization and writes results/plots to `outputs/`.

## Run

```bash
pip install -r requirements.txt
python main.py
```

NSGA-II runs with population 400 for 600 generations, fully vectorized (SBX
crossover, polynomial mutation, and tournament selection all operate on the
whole population/gene matrix at once, driven by a single seeded
`numpy.random.Generator`) — a run takes ~30s and is **byte-for-byte
reproducible** given the same seed (default `seed=7` in `main.py`,
market data seed `42` in `data.py`).

## Outputs (in `outputs/`)

- `pareto_front.csv` — every non-dominated portfolio found (objectives + weights).
- `representative_portfolios.csv` — Max Return, Min Risk, Max Diversification,
  Max Sharpe-like, and a "balanced" knee-point portfolio, with full weight breakdowns.
- `pareto_front_2d.png` / `pareto_front_3d.png` — visualizations of the trade-off surface.
- `convergence.png` — best objective values per generation.
