from __future__ import annotations
import numpy as np
from typing import Optional, Literal
import heapq

Method = Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"]
VALID_METHODS = set(Method.__args__)
REDUCIBLE_METHODS = {"single", "complete", "average", "weighted", "ward"}

def _validate_inputs(
    y: np.ndarray,
    method: str,
    min_cluster_size: Optional[int],
    max_cluster_size: Optional[int],
    min_penalty_weight: float,
    max_penalty_weight: float
) -> None:
    """Validate top-level API arguments.

    Ensures:
      - `method` is one of VALID_METHODS.
      - min/max cluster sizes are positive (if provided) and consistent.
      - penalty weights are non-negative.

    Raises
    ------
    ValueError / TypeError
        If any input is invalid.
    """
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
    """Return True iff `y` is a square 2-D array (n x n)."""
    return y.ndim == 2 and y.shape[0] == y.shape[1]

def _n_from_condensed_len(m: int) -> int:
    """Infer n from a condensed-length m = n*(n-1)/2; error if not triangular."""
    n = (1 + int(np.sqrt(1 + 8*m))) // 2
    if n*(n-1)//2 != m:
        raise ValueError("Invalid length for condensed distances.")
    return n

def _to_square(y: np.ndarray) -> np.ndarray:
    """Accept condensed 1-D or square 2-D distances and 
    return a symmetric (n,n) float matrix from either condensed 1-D or square 2-D.

    - If `y` is (n,n), copies to float, zeros the diagonal, and symmetrizes.
    - If `y` is 1-D condensed, expands to (n,n).
    """
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
    """Normalize/validate the constraint matrix.

    - None -> returns zeros((n,n)).
    - Must be (n,n) numeric; diagonal forced to 0; symmetrized (M+Mᵀ)/2.
    """
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
    """Lance–Williams update for merging a=(sa), b=(sb) when measuring to k=(sk).

    Parameters are pairwise base distances (dak, dbk, dab) and cluster sizes.
    Implements: single, complete, average (UPGMA), weighted (WPGMA),
    centroid (UPGMC), median (WPGMC), and ward (minimum variance).
    Returns a non-negative distance (clamped at 0).
    """
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
  

def _sort_and_label(Z_alg: np.ndarray, n: int) -> np.ndarray:
    """
    Given Z in *algorithm order* with first two cols containing fastcluster-style ids
    (leaves 0..n-1, then n..2n-2 as merges happen), return a copy that is
    (1) sorted by height (col 2) with a stable mergesort, and
    (2) relabeled to SciPy’s canonical scheme for that sorted order.
    """
    order = np.argsort(Z_alg[:, 2], kind="mergesort")
    Z_out = np.empty_like(Z_alg)

    # map from *old* node id (fastcluster order) -> *new* node id (sorted order)
    old_to_new = {i: i for i in range(n)}  # leaves map to themselves

    for r, s in enumerate(order):
        a_old = int(Z_alg[s, 0])
        b_old = int(Z_alg[s, 1])
        # children must already be mapped (heights are nondecreasing)
        a_new = old_to_new[a_old]
        b_new = old_to_new[b_old]
        if a_new > b_new:
            a_new, b_new = b_new, a_new

        Z_out[r, 0] = a_new
        Z_out[r, 1] = b_new
        Z_out[r, 2] = Z_alg[s, 2]  # height unchanged
        Z_out[r, 3] = Z_alg[s, 3]  # size unchanged

        # the parent created at *original* step s had id (n + s)
        # assign its *new* id as (n + r) in the sorted order
        old_to_new[n + s] = n + r

    return Z_out

def nn_chain_constrained(
    D: np.ndarray,
    P: np.ndarray,
    method: Literal["single", "complete", "average", "weighted", "ward"] = "single",
    *,
    min_cluster_size: Optional[int],
    max_cluster_size: Optional[int],
    min_penalty_weight: float,
    max_penalty_weight: float,
) -> np.ndarray:
    """
    Nearest-Neighbor Chain hierarchical clustering with soft constraints.

    Parameters
    ----------
    D : (n,n) float ndarray
        Symmetric base distance matrix. Diagonal ignored.
    P : (n,n) float ndarray
        Constraint penalty matrix. Negative entries encourage merges, positive entries
        discourage them. During clustering, penalties are accumulated:
        P[new,k] = P[i,k] + P[j,k].
    method : {'single','complete','average','weighted','ward'}, default='single'
        Linkage method. Must be reducible (NN-chain assumes reducibility).
    min_cluster_size, max_cluster_size : int or None
        Soft bounds on merged cluster sizes. Used only if the corresponding
        penalty weight is > 0.
    min_penalty_weight, max_penalty_weight : float
        Penalty weights for size constraints. Units match D unless distances
        are normalized.

    Returns
    -------
    Z : (n-1, 4) float ndarray
        SciPy-compatible linkage matrix. Each row: [idx_a, idx_b, dist, size].

    Notes
    -----
    - Equivalent to SciPy's NN-chain implementation if all penalties are zero.
    - Complexity O(n^2) for reducible methods.
    """
    n = D.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    sizes = np.ones(n, dtype=int)     # 0 means dead
    labels = list(range(n))           # fastcluster-style ids we write into Z (algorithm order)
    Z = np.zeros((n - 1, 4), dtype=float)

    chain: list[int] = []
    next_id = n
    step = 0

    while step < n - 1:
        # seed chain with the first live cluster if empty
        if not chain:
            for i in range(n):
                if sizes[i] > 0:
                    chain.append(i)
                    break

        # extend chain until reciprocal nearest neighbors
        while True:
            x = chain[-1]

            # prefer previous element in the chain (strict '<' keeps it on ties)
            if len(chain) > 1:
                y = chain[-2]
                best_val = (
                    D[x, y]
                    + P[x, y]
                    + _size_penalty(sizes[x], sizes[y],
                                    min_cluster_size, max_cluster_size,
                                    min_penalty_weight, max_penalty_weight)
                )
            else:
                y = -1
                best_val = np.inf

            # scan all live clusters in increasing index order
            for j in range(n):
                if j == x or sizes[j] == 0:
                    continue
                val = (
                    D[x, j]
                    + P[x, j]
                    + _size_penalty(sizes[x], sizes[j],
                                    min_cluster_size, max_cluster_size,
                                    min_penalty_weight, max_penalty_weight)
                )
                if val < best_val:   # STRICT: preserve previous neighbor on ties
                    best_val = val
                    y = j

            # reciprocal nearest neighbors?
            if len(chain) > 1 and y == chain[-2]:
                break
            chain.append(y)

        # pop last two, merge as (a,b) with a < b (fastcluster/SciPy convention)
        a = chain.pop()
        b = chain.pop()
        if a > b:
            a, b = b, a

        sa, sb = sizes[a], sizes[b]

        # record the merge in *algorithm order* (fastcluster ids in cols 0–1)
        Z[step, 0] = labels[a]
        Z[step, 1] = labels[b]
        Z[step, 2] = best_val if best_val > 0.0 else 0.0
        Z[step, 3] = sa + sb

        # mark a dead; b becomes the merged cluster
        sizes[a] = 0
        sizes[b] = sa + sb
        labels[b] = next_id
        next_id += 1
        step += 1

        # update base distances with Lance–Williams; penalties sum
        dab = D[a, b]  # base (unpenalized) distance used in LW update
        for k in range(n):
            if sizes[k] == 0 or k == b:
                continue
            dak, dbk = D[a, k], D[b, k]
            newd = _lw_update(method, sa, sb, sizes[k], dak, dbk, dab)
            if newd < 0.0:
                newd = 0.0
            D[b, k] = D[k, b] = newd

            # penalties add under merges (soft-constraint accumulation)
            P[b, k] = P[k, b] = P[a, k] + P[b, k]

    Z = _sort_and_label(Z, n)
    return Z

def _quadratic_linkage(
    D: np.ndarray,
    P: np.ndarray,
    method: Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"] = "single",
    *,
    min_cluster_size: Optional[int],
    max_cluster_size: Optional[int],
    min_penalty_weight: float,
    max_penalty_weight: float,
) -> np.ndarray:
    """
    Quadratic O(n^3) linkage.

    Brute-force baseline: at each step, scan all pairs to find the
    minimum penalized distance.

    Returns
    -------
    Z : (n-1, 4) float ndarray
    """
    n = D.shape[0]
    labels = list(range(n))
    sizes = np.ones(n, dtype=int)
    Z = np.zeros((n - 1, 4), dtype=float)
    next_id = n

    def adjusted(i: int, j: int) -> float:
        si, sj = sizes[i], sizes[j]
        base = D[i, j]
        pen = P[i, j] + _size_penalty(si, sj,
                                      min_cluster_size, max_cluster_size,
                                      min_penalty_weight, max_penalty_weight)
        return max(base + pen, 0.0)

    for step in range(n - 1):
        m = len(labels)
        best_i = best_j = -1
        best_val = np.inf
        for i in range(m - 1):
            for j in range(i + 1, m):
                val = adjusted(i, j)
                # tie-break lexicographically on original labels
                if (val < best_val - 1e-15 or
                    (abs(val - best_val) <= 1e-15 and
                     tuple(sorted((labels[i], labels[j]))) <
                     tuple(sorted((labels[best_i], labels[best_j]))) )):
                    best_val, best_i, best_j = val, i, j

        i, j = best_i, best_j
        if i > j:
            i, j = j, i

        Zi, Zj = labels[i], labels[j]
        si, sj = sizes[i], sizes[j]
        Z[step, 0] = Zi
        Z[step, 1] = Zj
        Z[step, 2] = max(best_val, 0.0)
        Z[step, 3] = si + sj

        new_base, new_pen = [], []
        for k in range(len(labels)):
            if k == i or k == j:
                continue
            sk = sizes[k]
            dak, dbk, dab = D[i, k], D[j, k], D[i, j]
            new_base.append(_lw_update(method, si, sj, sk, dak, dbk, dab))
            new_pen.append(P[i, k] + P[j, k])

        keep = [k for k in range(len(labels)) if k not in (i, j)]
        D = D[np.ix_(keep, keep)]
        P = P[np.ix_(keep, keep)]
        sizes = sizes[keep]
        labels = [labels[k] for k in keep]

        if len(labels) > 0:
            nb = np.maximum(np.asarray(new_base, float), 0.0)
            D = np.pad(D, ((0, 1), (0, 1)), constant_values=0.0)
            P = np.pad(P, ((0, 1), (0, 1)), constant_values=0.0)
            D[-1, :-1] = D[:-1, -1] = nb
            P[-1, :-1] = P[:-1, -1] = np.asarray(new_pen, float)

        sizes = np.append(sizes, si + sj)
        labels.append(next_id)
        next_id += 1

    return Z

def _fast_linkage_lazy(
    D: np.ndarray,
    P: np.ndarray,
    method: Literal["centroid","median"],
    *,
    min_cluster_size=None,
    max_cluster_size=None,
    min_penalty_weight=0.0,
    max_penalty_weight=0.0,
):
    """
    Lazy-heap HAC with soft constraints (generic fallback).

    Uses a global priority queue (min-heap) to find the next merge.
    Each candidate distance is checked lazily and recomputed if stale.
    Correct but may be slower than NN-chain for reducible methods.

    Returns
    -------
    Z : (n-1, 4) float ndarray
        Linkage matrix in algorithm order (caller should sort/label).
    """
    n = D.shape[0]
    sizes = np.ones(n, dtype=int)
    labels = list(range(n))
    alive = [True] * n
    Z = np.zeros((n - 1, 4), dtype=float)
    next_id = n

    def adjusted(i, j):
        si, sj = sizes[i], sizes[j]
        base = D[i, j]
        pen = (
            P[i, j]
            + _size_penalty(si, sj,
                            min_cluster_size, max_cluster_size,
                            min_penalty_weight, max_penalty_weight)
        )
        return max(base + pen, 0.0)

    # build initial heap
    heap = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            val = adjusted(i, j)
            heapq.heappush(heap, (val, i, j))

    step = 0
    while step < n - 1:
        # pop until valid
        while True:
            val, i, j = heapq.heappop(heap)
            if not alive[i] or not alive[j]:
                continue  # dead cluster
            new_val = adjusted(i, j)
            if new_val > val + 1e-12:  # stale
                heapq.heappush(heap, (new_val, i, j))
                continue
            # good pair
            break

        if i > j:
            i, j = j, i

        si, sj = sizes[i], sizes[j]

        # record merge
        Z[step, 0] = labels[i]
        Z[step, 1] = labels[j]
        Z[step, 2] = new_val
        Z[step, 3] = si + sj

        # update structures
        alive[i] = False
        sizes[j] = si + sj
        labels[j] = next_id
        next_id += 1
        step += 1

        # update distances to new cluster j
        dab = D[i, j]
        for k in range(n):
            if not alive[k] or k == j:
                continue
            dak, dbk = D[i, k], D[j, k]
            newd = _lw_update(method, si, sj, sizes[k], dak, dbk, dab)
            if newd < 0.0:
                newd = 0.0
            D[j, k] = D[k, j] = newd
            P[j, k] = P[k, j] = P[i, k] + P[j, k]
            heapq.heappush(heap, (adjusted(j, k), j, k))

    return Z


def constrained_linkage(
    y: np.ndarray,
    method: Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"] = "single",
    *,
    min_cluster_size: Optional[int] = None,
    max_cluster_size: Optional[int] = None,
    min_penalty_weight: float = 0.0,
    max_penalty_weight: float = 0.0,
    constraint_matrix: Optional[np.ndarray] = None,
    normalize_distances: bool = False,
) -> np.ndarray:
    """
    Constrained hierarchical linkage (SciPy-compatible Z).
    Uses NN-chain for reducible methods, quadratic fallback otherwise.

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

    _validate_inputs(
        y, method, min_cluster_size, max_cluster_size,
        min_penalty_weight, max_penalty_weight
    )

    # --- preprocess distances & penalties exactly once ---
    D = _to_square(np.asarray(y))
    if normalize_distances:
        mx = D.max()
        if mx > 0:
            D = D / mx
    P = _validate_constraint_matrix(constraint_matrix, D.shape[0])

    # Choose implementation function
    impl = nn_chain_constrained if method in REDUCIBLE_METHODS else _fast_linkage_lazy
    # Shared keyword args
    kwargs = dict(
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        min_penalty_weight=min_penalty_weight,
        max_penalty_weight=max_penalty_weight,
    )

    return impl(D, P, method, **kwargs)