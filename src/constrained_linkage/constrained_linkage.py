from __future__ import annotations
import numpy as np
from typing import Optional, Literal

Method = Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"]
VALID_METHODS = set(Method.__args__)

def _validate_inputs(
    y: np.ndarray,
    method: str,
    min_cluster_size: Optional[int],
    max_cluster_size: Optional[int],
    min_penalty_weight: float,
    max_penalty_weight: float
) -> None:
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method: {method!r}. Must be one of {sorted(VALID_METHODS)}")
    if min_cluster_size is not None and min_cluster_size < 1:
        raise ValueError("min_cluster_size must be >= 1")
    if max_cluster_size is not None and max_cluster_size < 1:
        raise ValueError("max_cluster_size must be >= 1")
    if (min_cluster_size is not None and max_cluster_size is not None and min_cluster_size > max_cluster_size): 
        raise ValueError("min_cluster_size cannot be greater than max_cluster_size")
    if min_penalty_weight < 0 or max_penalty_weight < 0:
        raise ValueError("Penalty weights must be non-negative")

def _is_square(y: np.ndarray) -> bool:
    return y.ndim == 2 and y.shape[0] == y.shape[1]

def _n_from_condensed_len(m: int) -> int:
    n = (1 + int(np.sqrt(1 + 8*m))) // 2
    if n*(n-1)//2 != m:
        raise ValueError("Invalid length for condensed distances.")
    return n

def _to_square(y: np.ndarray) -> np.ndarray:
    """Accept condensed 1-D or square 2-D distances and return a symmetric square matrix."""
    y = np.asarray(y)
    if _is_square(y):
        D = y.astype(float, copy=True)
        if D.shape[0] < 2:
            raise ValueError("Need at least 2 observations.")
        np.fill_diagonal(D, 0.0)
        return (D + D.T) / 2.0
    if y.ndim != 1:
        raise ValueError("Distance input must be condensed 1-D or square 2-D.")
    n = _n_from_condensed_len(y.shape[0])
    D = np.zeros((n, n), dtype=float)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            v = float(y[k]); k += 1
            D[i, j] = D[j, i] = v
    return D

def _validate_constraint_matrix(M: Optional[np.ndarray], n: int) -> np.ndarray:
    if M is None:
        return np.zeros((n, n), dtype=float)
    if not isinstance(M, np.ndarray):
        raise TypeError("constraint_matrix must be a NumPy ndarray")
    if M.shape != (n, n):
        raise ValueError(f"constraint_matrix must be shape {(n, n)}, got {M.shape}.")
    if not np.issubdtype(M.dtype, np.number):
        raise TypeError("constraint_matrix must have a numeric dtype")
    np.fill_diagonal(M, 0.0)
    return (M + M.T) / 2.0

def _size_penalty(
    sa: int, sb: int,
    cmin: Optional[int], cmax: Optional[int],
    wmin: float, wmax: float
) -> float:
    """
    Soft size penalties:
      - Min size: encourage merging clusters that are individually below cmin
                  by subtracting wmin * deficit_a/b (negative penalty).
      - Max size: discourage merges whose *result* would exceed cmax
                  by adding wmax * (sa+sb - cmax) (positive penalty).

    Assumes wmin,wmax >= 0. With normalize_distances=True, penalties are
    scale-free relative to the [0,1] normalized distance range.
    """
    pen = 0.0

    # Encourage merging under-min clusters (acts even when cmin=2)
    if cmin is not None:
        deficit_a = max(0, cmin - sa)
        deficit_b = max(0, cmin - sb)
        pen -= wmin * float(deficit_a + deficit_b)

    # Discourage creating over-max clusters (post-merge)
    if cmax is not None:
        excess = max(0, (sa + sb) - cmax)
        pen += wmax * float(excess)

    return pen

def _lw_update(method: Method,
               sa: int, sb: int, sk: int,
               dak: float, dbk: float, dab: float) -> float:
    # Lance–Williams updates. Use squared internals where needed; sqrt at the end.
    if method == "single":
        return min(dak, dbk)
    if method == "complete":
        return max(dak, dbk)
    if method == "average":  # UPGMA
        return (sa * dak + sb * dbk) / (sa + sb)
    if method == "weighted":  # WPGMA
        return 0.5 * (dak + dbk)
    if method == "centroid":  # UPGMC
        sa_sb = sa + sb
        val2 = (sa/sa_sb) * (dak**2) + (sb/sa_sb) * (dbk**2) - (sa*sb)/(sa_sb**2) * (dab**2)
        return np.sqrt(max(val2, 0.0))
    if method == "median":  # WPGMC
        val2 = 0.5*(dak**2 + dbk**2) - 0.25*(dab**2)
        return np.sqrt(max(val2, 0.0))
    if method == "ward":  # Minimum variance (Ward.D)
        sa_sb = sa + sb
        total = sa_sb + sk
        val2 = ((sa + sk)/total) * (dak**2) + ((sb + sk)/total) * (dbk**2) - (sk/total) * (dab**2)
        return np.sqrt(max(val2, 0.0))
    raise ValueError(f"Unknown method {method!r}")

def constrained_linkage(
    y: np.ndarray,
    method: Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"] = "single",
    *,
    min_cluster_size: Optional[int] = None,
    max_cluster_size: Optional[int] = None,
    min_penalty_weight: float = 0.0,
    max_penalty_weight: float = 0.0,
    constraint_matrix: Optional[np.ndarray] = None,
    normalize_distances: bool = False
) -> np.ndarray:
    """
    Constrained hierarchical linkage (SciPy-compatible Z) with soft size penalties
    and pairwise constraint-matrix penalties. NumPy-only.

    Parameters
    ----------
    y : array
        Either condensed 1-D distances (len n*(n-1)/2) or an (n,n) distance matrix.
    method : {'single','complete','average','weighted','centroid','median','ward'}
        Linkage rule (Lance–Williams).
    min_cluster_size, max_cluster_size : int or None
        Soft bounds on merged sizes. Set a weight to activate.
    min_penalty_weight, max_penalty_weight : float
        Weights for encouraging under-min merges / discouraging over-max merges.
        Units match the distance scale (set normalize_distances=True to make these unitless).
    constraint_matrix : (n,n) array or None
        Pairwise penalties/rewards. Cluster–cluster penalties are summed over members
        but updated incrementally as merges happen: P[new,k] = P[i,k] + P[j,k].
        Negative = encourage; positive = discourage (soft constraints).
    normalize_distances : bool
        If True, divides all base distances by their max so penalty weights live in [0,1].

    Returns
    -------
    Z : (n-1,4) float ndarray
        SciPy-compatible linkage matrix with [idx_a, idx_b, dist, size].
        Leaves are 0..n-1, new clusters are n..2n-2.
    """
    _validate_inputs(y, method, min_cluster_size, max_cluster_size, min_penalty_weight, max_penalty_weight)

    # Base distances (square)
    D = _to_square(np.asarray(y))
    n = D.shape[0]
    if normalize_distances:
        mx = D.max()
        if mx > 0:
            D = D / mx

    # Penalties (square)
    P = _validate_constraint_matrix(constraint_matrix, n)

    # Active clusters
    labels = list(range(n))        # current cluster ids
    sizes = np.ones(n, dtype=int)
    Z = np.zeros((n - 1, 4), dtype=float)
    next_id = n

    def adjusted(i: int, j: int) -> float:
        si, sj = sizes[i], sizes[j]
        base = D[i, j]
        pen = P[i, j] + _size_penalty(si, sj, min_cluster_size, max_cluster_size,
                                      min_penalty_weight, max_penalty_weight)
        return max(base + pen, 0.0)

    for step in range(n - 1):
        m = len(labels)
        best_i = best_j = -1
        best_val = np.inf
        for i in range(m - 1):
            for j in range(i + 1, m):
                val = adjusted(i, j)
                # Primary: strictly smaller distance wins
                # Tie: lexicographic on original labels
                if (
                    val < best_val - 1e-15 or
                    (abs(val - best_val) <= 1e-15 and
                    tuple(sorted((labels[i], labels[j]))) <
                    tuple(sorted((labels[best_i], labels[best_j]))))
                ):
                    best_val, best_i, best_j = val, i, j

        i, j = best_i, best_j
        if i > j:
            i, j = j, i

        # Record merge with original ids
        Zi, Zj = labels[i], labels[j]
        si, sj = sizes[i], sizes[j]
        Z[step, 0] = Zi
        Z[step, 1] = Zj
        Z[step, 2] = max(best_val, 0.0)
        Z[step, 3] = si + sj

        # Build new distances/penalties to other clusters via Lance–Williams; penalties sum
        new_base = []
        new_pen = []
        for k in range(len(labels)):
            if k == i or k == j:
                continue
            sk = sizes[k]
            dak = D[i, k]
            dbk = D[j, k]
            dab = D[i, j]
            new_base.append(_lw_update(method, si, sj, sk, dak, dbk, dab))
            new_pen.append(P[i, k] + P[j, k])

        # Remove j then i; append new
        keep = [k for k in range(len(labels)) if k not in (i, j)]
        D = D[np.ix_(keep, keep)]
        P = P[np.ix_(keep, keep)]
        sizes = sizes[keep]
        labels = [labels[k] for k in keep]

        if len(labels) > 0:
            nb = np.array(new_base, dtype=float)
            np.clip(nb, 0.0, np.inf, out=nb)
            # expand D
            D = np.pad(D, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
            D[-1, :-1] = D[:-1, -1] = nb
            # expand P
            np_pen = np.array(new_pen, dtype=float)
            P = np.pad(P, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
            P[-1, :-1] = P[:-1, -1] = np_pen

        sizes = np.append(sizes, si + sj)
        labels.append(next_id)
        next_id += 1

    return Z