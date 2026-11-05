"""Multi-Objective Portfolio Optimization Using Evolutionary Algorithms.

Runs NSGA-II over a synthetic multi-asset universe to find the Pareto-optimal
set of portfolios trading off expected return, risk, and diversification,
then reports and plots representative solutions.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import ASSETS, N_ASSETS, generate_returns, annualized_stats
from nsga2 import run_nsga2
from portfolio import MAX_WEIGHT, make_evaluator, make_repair, portfolio_metrics

OUT = "outputs"


def pick_knee_point(objs: np.ndarray) -> int:
    """Index of the solution closest to the ideal point in normalized space."""
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    span = np.where(maxs - mins > 1e-12, maxs - mins, 1.0)
    norm = (objs - mins) / span
    dist = np.linalg.norm(norm, axis=1)
    return int(np.argmin(dist))


def summarize(label: str, w: np.ndarray, mu, cov) -> dict:
    m = portfolio_metrics(w, mu, cov)
    row = {"portfolio": label, **m}
    for name, weight in zip(ASSETS, w):
        row[name] = weight
    return row


def main():
    print("=" * 70)
    print("Multi-Objective Portfolio Optimization Using Evolutionary Algorithms")
    print("=" * 70)

    print(f"\nSimulating market data for {N_ASSETS} assets...")
    returns = generate_returns()
    mu, cov = annualized_stats(returns)
    returns.to_csv(f"{OUT}/daily_returns.csv")

    pop_size, n_gen, seed = 400, 600, 7
    print(f"Running NSGA-II (pop={pop_size}, gens={n_gen}, max position weight={MAX_WEIGHT}, seed={seed})...\n")
    evaluate = make_evaluator(mu, cov)
    repair = make_repair(MAX_WEIGHT)
    result = run_nsga2(
        evaluate=evaluate,
        repair=repair,
        n_var=N_ASSETS,
        low=np.zeros(N_ASSETS),
        high=np.full(N_ASSETS, MAX_WEIGHT),
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
    )

    pareto_idx = result.fronts[0]
    pareto_pop = result.population[pareto_idx]
    pareto_objs = result.objectives[pareto_idx]

    # Deduplicate near-identical solutions on the front
    rounded = np.round(pareto_objs, 5)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    pareto_pop = pareto_pop[unique_idx]
    pareto_objs = pareto_objs[unique_idx]

    print(f"\nPareto front found: {len(pareto_pop)} non-dominated portfolios")

    front_df = pd.DataFrame(
        {
            "expected_return": -pareto_objs[:, 0],
            "risk_std": pareto_objs[:, 1],
            "hhi": pareto_objs[:, 2],
            "effective_n_assets": 1.0 / pareto_objs[:, 2],
        }
    )
    weights_df = pd.DataFrame(pareto_pop, columns=ASSETS)
    full_front = pd.concat([front_df, weights_df], axis=1).sort_values(
        "expected_return"
    )
    full_front.to_csv(f"{OUT}/pareto_front.csv", index=False)

    max_ret_i = np.argmin(pareto_objs[:, 0])
    min_risk_i = np.argmin(pareto_objs[:, 1])
    max_div_i = np.argmin(pareto_objs[:, 2])
    knee_i = pick_knee_point(pareto_objs)
    sharpe = -pareto_objs[:, 0] / np.maximum(pareto_objs[:, 1], 1e-9)
    max_sharpe_i = int(np.argmax(sharpe))

    rows = [
        summarize("Max Return", pareto_pop[max_ret_i], mu, cov),
        summarize("Min Risk", pareto_pop[min_risk_i], mu, cov),
        summarize("Max Diversification", pareto_pop[max_div_i], mu, cov),
        summarize("Balanced (knee point)", pareto_pop[knee_i], mu, cov),
        summarize("Max Sharpe-like", pareto_pop[max_sharpe_i], mu, cov),
    ]
    summary_df = pd.DataFrame(rows).set_index("portfolio")
    summary_df.to_csv(f"{OUT}/representative_portfolios.csv")

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    print("\nRepresentative portfolios (objective metrics):")
    print(summary_df[["return", "risk", "sharpe_like", "hhi", "effective_n_assets"]])

    print("\nRepresentative portfolios (weights):")
    print(summary_df[ASSETS])

    # --- Plots ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        pareto_objs[:, 1],
        -pareto_objs[:, 0],
        c=1.0 / pareto_objs[:, 2],
        cmap="viridis",
        s=40,
        edgecolor="k",
        linewidth=0.3,
    )
    for i, label in zip(
        [max_ret_i, min_risk_i, max_div_i, knee_i, max_sharpe_i],
        ["Max Return", "Min Risk", "Max Diversification", "Balanced", "Max Sharpe"],
    ):
        ax.scatter(
            pareto_objs[i, 1], -pareto_objs[i, 0], marker="*", s=300,
            edgecolor="black", linewidth=1.0, zorder=5,
        )
        ax.annotate(label, (pareto_objs[i, 1], -pareto_objs[i, 0]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Effective number of assets (1/HHI)")
    ax.set_xlabel("Risk (annualized std. dev.)")
    ax.set_ylabel("Expected annual return")
    ax.set_title("Pareto Front: Return vs. Risk, colored by Diversification")
    fig.tight_layout()
    fig.savefig(f"{OUT}/pareto_front_2d.png", dpi=150)

    fig3d = plt.figure(figsize=(9, 7))
    ax3d = fig3d.add_subplot(projection="3d")
    ax3d.scatter(
        pareto_objs[:, 1], -pareto_objs[:, 0], 1.0 / pareto_objs[:, 2],
        c=-pareto_objs[:, 0], cmap="plasma", s=35,
    )
    ax3d.set_xlabel("Risk (std dev)")
    ax3d.set_ylabel("Expected return")
    ax3d.set_zlabel("Effective N assets")
    ax3d.set_title("3D Pareto Front")
    fig3d.tight_layout()
    fig3d.savefig(f"{OUT}/pareto_front_3d.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    history = np.array(result.history_best)
    ax2.plot(-history[:, 0], label="best return (max over pop)")
    ax2.plot(history[:, 1], label="best risk (min over pop)")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Objective value")
    ax2.set_title("Convergence over generations")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{OUT}/convergence.png", dpi=150)

    print(f"\nSaved outputs to ./{OUT}/:")
    print("  pareto_front.csv, representative_portfolios.csv")
    print("  pareto_front_2d.png, pareto_front_3d.png, convergence.png")


if __name__ == "__main__":
    main()
