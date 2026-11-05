"""Generic NSGA-II (Non-dominated Sorting Genetic Algorithm II).

Minimizes an arbitrary number of objectives. Pluggable via an
``evaluate(pop) -> (n, n_obj)`` function and a ``repair(pop) -> pop``
function so it can be reused for problems other than portfolios.

Fully vectorized (no per-individual/per-gene Python loops) and driven
entirely by a single seeded ``numpy.random.Generator`` so runs are
exactly reproducible given the same seed.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def fast_non_dominated_sort(objs: np.ndarray) -> list[np.ndarray]:
    """Return list of index arrays, one per Pareto front (front[0] = best)."""
    n = objs.shape[0]
    dominates = np.zeros((n, n), dtype=bool)
    for i in range(n):
        # i dominates j if i is <= j on all objectives and < on at least one
        le = np.all(objs[i] <= objs, axis=1)
        lt = np.any(objs[i] < objs, axis=1)
        dominates[i] = le & lt

    domination_count = dominates.sum(axis=0)  # how many dominate j
    fronts = []
    remaining = np.ones(n, dtype=bool)
    current_rank = domination_count.copy()

    while remaining.any():
        front_mask = remaining & (current_rank == 0)
        if not front_mask.any():
            # numerical edge case guard
            front_mask = remaining
        front = np.where(front_mask)[0]
        fronts.append(front)
        remaining[front] = False
        for i in front:
            dominated_by_i = dominates[i] & remaining
            current_rank[dominated_by_i] -= 1
    return fronts


def crowding_distance(objs: np.ndarray) -> np.ndarray:
    n, m = objs.shape
    dist = np.zeros(n)
    for k in range(m):
        order = np.argsort(objs[:, k])
        sorted_vals = objs[order, k]
        span = sorted_vals[-1] - sorted_vals[0]
        if span <= 1e-12:
            continue
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        dist[order[1:-1]] += (sorted_vals[2:] - sorted_vals[:-2]) / span
    return dist


def tournament_select_batch(ranks, crowd, n_select, rng, k=2):
    """Vectorized binary tournament selection -> array of n_select winners."""
    n = len(ranks)
    pool = rng.integers(0, n, size=(n_select, k))
    r0, r1 = ranks[pool[:, 0]], ranks[pool[:, 1]]
    c0, c1 = crowd[pool[:, 0]], crowd[pool[:, 1]]
    first_wins = (r0 < r1) | ((r0 == r1) & (c0 > c1))
    return np.where(first_wins, pool[:, 0], pool[:, 1])


def sbx_crossover_batch(P1, P2, low, high, eta=15.0, prob=0.9, rng=None):
    """Vectorized simulated binary crossover over a whole batch of pairs."""
    rng = rng or np.random.default_rng()
    m, n = P1.shape
    eps = 1e-14

    x1, x2 = P1, P2
    lo = np.broadcast_to(low, (m, n))
    hi = np.broadcast_to(high, (m, n))
    x1_ = np.minimum(x1, x2)
    x2_ = np.maximum(x1, x2)
    diff = x2_ - x1_
    safe_diff = np.where(diff > eps, diff, 1.0)

    rand = rng.random((m, n))

    beta1 = 1.0 + 2.0 * (x1_ - lo) / safe_diff
    alpha1 = 2.0 - beta1 ** (-(eta + 1.0))
    betaq1 = np.where(
        rand <= 1.0 / alpha1,
        (rand * alpha1) ** (1.0 / (eta + 1.0)),
        (1.0 / (2.0 - rand * alpha1)) ** (1.0 / (eta + 1.0)),
    )
    child1 = 0.5 * ((x1_ + x2_) - betaq1 * diff)

    beta2 = 1.0 + 2.0 * (hi - x2_) / safe_diff
    alpha2 = 2.0 - beta2 ** (-(eta + 1.0))
    betaq2 = np.where(
        rand <= 1.0 / alpha2,
        (rand * alpha2) ** (1.0 / (eta + 1.0)),
        (1.0 / (2.0 - rand * alpha2)) ** (1.0 / (eta + 1.0)),
    )
    child2 = 0.5 * ((x1_ + x2_) + betaq2 * diff)

    valid = diff > eps
    c1 = np.where(valid, np.clip(child1, lo, hi), x1)
    c2 = np.where(valid, np.clip(child2, lo, hi), x2)

    do_cross = (rng.random(m) <= prob)[:, None]
    c1 = np.where(do_cross, c1, x1)
    c2 = np.where(do_cross, c2, x2)
    return c1, c2


def polynomial_mutation_batch(X, low, high, eta=20.0, prob=None, rng=None):
    """Vectorized polynomial mutation over a whole population matrix."""
    rng = rng or np.random.default_rng()
    m, n = X.shape
    prob = prob if prob is not None else 1.0 / n
    lo = np.broadcast_to(low, (m, n))
    hi = np.broadcast_to(high, (m, n))
    span = np.where((hi - lo) > 1e-14, hi - lo, 1.0)

    do_mut = rng.random((m, n)) <= prob
    r = rng.random((m, n))
    mut_pow = 1.0 / (eta + 1.0)

    delta1 = (X - lo) / span
    delta2 = (hi - X) / span

    xy1 = 1.0 - delta1
    val1 = 2.0 * r + (1.0 - 2.0 * r) * (xy1 ** (eta + 1.0))
    deltaq1 = val1 ** mut_pow - 1.0

    xy2 = 1.0 - delta2
    val2 = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy2 ** (eta + 1.0))
    deltaq2 = 1.0 - val2 ** mut_pow

    deltaq = np.where(r < 0.5, deltaq1, deltaq2)
    mutated = np.clip(X + deltaq * span, lo, hi)
    return np.where(do_mut, mutated, X)


@dataclass
class NSGA2Result:
    population: np.ndarray
    objectives: np.ndarray
    fronts: list = field(default_factory=list)
    history_best: list = field(default_factory=list)


def run_nsga2(
    evaluate: Callable[[np.ndarray], np.ndarray],
    repair: Callable[[np.ndarray], np.ndarray],
    n_var: int,
    low: np.ndarray,
    high: np.ndarray,
    pop_size: int = 150,
    n_gen: int = 250,
    seed: int = 1,
    crossover_eta: float = 15.0,
    mutation_eta: float = 20.0,
    verbose: bool = True,
) -> NSGA2Result:
    if pop_size % 2 != 0:
        raise ValueError("pop_size must be even")

    rng = np.random.default_rng(seed)

    pop = rng.uniform(low, high, size=(pop_size, n_var))
    pop = repair(pop)
    objs = evaluate(pop)
    history = []

    n_pairs = pop_size // 2

    for gen in range(n_gen):
        fronts = fast_non_dominated_sort(objs)
        ranks = np.empty(pop_size, dtype=int)
        crowd = np.empty(pop_size, dtype=float)
        for r, front in enumerate(fronts):
            ranks[front] = r
            crowd[front] = crowding_distance(objs[front])

        parents1 = tournament_select_batch(ranks, crowd, n_pairs, rng)
        parents2 = tournament_select_batch(ranks, crowd, n_pairs, rng)

        c1, c2 = sbx_crossover_batch(
            pop[parents1], pop[parents2], low, high, eta=crossover_eta, rng=rng
        )
        c1 = polynomial_mutation_batch(c1, low, high, eta=mutation_eta, rng=rng)
        c2 = polynomial_mutation_batch(c2, low, high, eta=mutation_eta, rng=rng)
        offspring = np.vstack([c1, c2])

        offspring = repair(offspring)
        off_objs = evaluate(offspring)

        combined_pop = np.vstack([pop, offspring])
        combined_objs = np.vstack([objs, off_objs])

        fronts = fast_non_dominated_sort(combined_objs)
        new_indices = []
        for front in fronts:
            if len(new_indices) + len(front) <= pop_size:
                new_indices.extend(front.tolist())
            else:
                remaining_slots = pop_size - len(new_indices)
                cd = crowding_distance(combined_objs[front])
                order = np.argsort(-cd)
                chosen = front[order[:remaining_slots]]
                new_indices.extend(chosen.tolist())
                break

        pop = combined_pop[new_indices]
        objs = combined_objs[new_indices]

        history.append(objs.min(axis=0).copy())
        if verbose and (gen % 25 == 0 or gen == n_gen - 1):
            print(
                f"gen {gen:4d} | best -return={objs[:,0].min():.4f} "
                f"best risk={objs[:,1].min():.4f} best HHI={objs[:,2].min():.4f}"
            )

    final_fronts = fast_non_dominated_sort(objs)
    return NSGA2Result(population=pop, objectives=objs, fronts=final_fronts, history_best=history)
