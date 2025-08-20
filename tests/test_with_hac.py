import numpy as np
import pytest
sp = pytest.importorskip("scipy")  # skip entire module if SciPy isn't installed

from constrained_linkage import constrained_linkage
from scipy.cluster import hierarchy as hierarchy
from scipy.spatial.distance import squareform

RANDOM_SEED = 42
# For constraint-behavior coverage: pick one from each family.
CONSTRAINT_METHODS = ["average", "centroid"]  # NN-chain + heap-fallback

def make_sines(n_groups=3, per_group=6, length=200, noise=0.05, random_state=0):
    rng = np.random.default_rng(random_state)
    X = []
    labels = []
    t = np.linspace(0, 2*np.pi, length)
    for g in range(n_groups):
        phase = rng.uniform(0, 2*np.pi)
        freq = 1.0 + 0.1 * g
        for _ in range(per_group):
            x = np.sin(freq * t + phase) + noise * rng.standard_normal(length)
            X.append(x)
            labels.append(g)
    return np.asarray(X), np.asarray(labels)

def euclidean_square(X: np.ndarray) -> np.ndarray:
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

def partitions_match(a: np.ndarray, b: np.ndarray) -> bool:
    # Compare partitions up to label permutation by pairwise co-membership.
    A = a[:, None] == a[None, :]
    B = b[:, None] == b[None, :]
    return np.all(A == B)

def test_unconstrained_matches_scipy_average_partition():
    X, _ = make_sines(n_groups=3, per_group=5, random_state=RANDOM_SEED)
    D = euclidean_square(X)
    # Our linkage
    Z_ours = constrained_linkage(D, method="average")
    # SciPy linkage on the same distances
    d_cond = squareform(D, checks=False)
    Z_sp = hierarchy.linkage(d_cond, method="average", optimal_ordering=False)

    # Compare partitions at k=3 clusters
    k = 3
    labels_ours = hierarchy.fcluster(Z_ours, k, criterion="maxclust")
    labels_sp = hierarchy.fcluster(Z_sp, k, criterion="maxclust")
    assert partitions_match(labels_ours, labels_sp)

@pytest.mark.parametrize("method", CONSTRAINT_METHODS)
def test_constraints_push_within_group_merges_first(method):
    X, true_g = make_sines(n_groups=3, per_group=4)
    D = euclidean_square(X)
    n = D.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if true_g[i] != true_g[j]:
                M[i, j] = 1.0
    Z_con = constrained_linkage(D, method=method, constraint_matrix=M)
    labels_con = hierarchy.fcluster(Z_con, 3, criterion="maxclust")
    assert partitions_match(labels_con, true_g)

@pytest.mark.parametrize("method", CONSTRAINT_METHODS)
def test_max_cluster_size_changes_partition(method):
    X = np.array([[0.00], [0.01], [0.02], [10.0], [20.0], [30.0]])
    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    # Baseline: SciPy (unconstrained) **with the same method**
    Z_sp = hierarchy.linkage(d_cond, method=method, optimal_ordering=False)
    labels_sp = hierarchy.fcluster(Z_sp, 3, criterion="maxclust")

    # Constrained: forbid clusters larger than 2
    Z_con = constrained_linkage(
        D, method=method,
        max_cluster_size=2, max_penalty_weight=0.2,
        normalize_distances=True
    )
    labels_con = hierarchy.fcluster(Z_con, 3, criterion="maxclust")

    assert not partitions_match(labels_sp, labels_con)

@pytest.mark.parametrize("method", CONSTRAINT_METHODS)
def test_constraint_matrix_should_and_shouldnot_link(method):
    X = np.array([[0.0], [0.1], [5.0], [10.0]])
    D = euclidean_square(X)
    n = D.shape[0]
    M = np.zeros((n, n))
    M[0, 3] = M[3, 0] = -0.5  # should-link
    M[0, 1] = M[1, 0] =  0.5  # should-not-link

    Z = constrained_linkage(
        D, method=method,
        constraint_matrix=M, normalize_distances=True
    )
    labels = hierarchy.fcluster(Z, 2, criterion="maxclust")
    assert labels[0] == labels[3]
    assert labels[0] != labels[1]

@pytest.mark.parametrize("method", CONSTRAINT_METHODS)
def test_max_penalty_enforces_equal_sized_clusters(method):
    X = np.array([[0.00],[0.01],[0.02],[0.03],[10.0],[20.0]])
    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    Z_sp = hierarchy.linkage(d_cond, method=method, optimal_ordering=False)
    labels_sp = hierarchy.fcluster(Z_sp, 3, criterion="maxclust")
    sizes_sp = sorted(np.bincount(labels_sp)[1:])

    Z_con = constrained_linkage(
        D, method=method,
        max_cluster_size=2, max_penalty_weight=0.5,
        normalize_distances=True
    )
    labels_con = hierarchy.fcluster(Z_con, 3, criterion="maxclust")
    sizes_con = sorted(np.bincount(labels_con)[1:])

    assert sizes_con == [2, 2, 2]
    assert sizes_sp != sizes_con

@pytest.mark.parametrize("method", CONSTRAINT_METHODS)
def test_min_size_constraint_reduces_singletons(method):
    X = np.array([[0.00],[0.01],[0.02],[5.0],[10.0],[15.0],[20.0],[25.0]])
    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    Z_sp = hierarchy.linkage(d_cond, method=method, optimal_ordering=False)
    labels_sp = hierarchy.fcluster(Z_sp, 4, criterion="maxclust")
    min_size_sp = min(sorted(np.bincount(labels_sp)[1:]))

    Z_con = constrained_linkage(
        D, method=method,
        min_cluster_size=3, min_penalty_weight=0.8,
        normalize_distances=True
    )
    labels_con = hierarchy.fcluster(Z_con, 4, criterion="maxclust")
    min_size_con = min(sorted(np.bincount(labels_con)[1:]))

    assert min_size_con >= min_size_sp
    assert min_size_con >= 2

@pytest.mark.parametrize("method", CONSTRAINT_METHODS)
def test_penalty_normalization_effect(method):
    X = np.array([[0.0], [1000.0], [2000.0]])
    D = euclidean_square(X)

    Z_no = constrained_linkage(
        D, method=method,
        max_cluster_size=2, max_penalty_weight=0.5,
        normalize_distances=False
    )
    Z_yes = constrained_linkage(
        D, method=method,
        max_cluster_size=2, max_penalty_weight=0.5,
        normalize_distances=True
    )
    assert not np.allclose(Z_no[:, 2], Z_yes[:, 2])

@pytest.mark.parametrize("method", ["single", "complete", "average", "weighted", "centroid", "median", "ward"])
def test_method_parity_partitions_with_scipy(method):
    # deterministic medium-sized dataset
    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.normal(size=(120, 1))

    # Euclidean square + condensed (SciPy expects condensed)
    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    # our linkage (no constraints)
    Z_ours = constrained_linkage(D, method=method)

    # SciPy linkage
    Z_sp = hierarchy.linkage(d_cond, method=method, optimal_ordering=False)

    # compare partitions at multiple k; partitions should be identical (up to relabeling)
    for k in (2, 3, 4, 5):
        labels_ours = hierarchy.fcluster(Z_ours, k, criterion="maxclust")
        labels_sp = hierarchy.fcluster(Z_sp, k, criterion="maxclust")
        assert partitions_match(labels_ours, labels_sp), f"partition mismatch for method={method}, k={k}"

@pytest.mark.parametrize("method", ["single", "complete", "average", "weighted", "centroid", "median", "ward"])
def test_method_parity_heights_close(method):
    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.normal(size=(40, 3))
    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    Z_ours = constrained_linkage(D, method=method)
    Z_sp = hierarchy.linkage(d_cond, method=method, optimal_ordering=False)

    # sort by height then by pair (stable compare)
    idx_ours = np.lexsort((Z_ours[:,0], Z_ours[:,1], Z_ours[:,2]))
    idx_sp   = np.lexsort((Z_sp[:,0],   Z_sp[:,1],   Z_sp[:,2]))

    # heights only
    h_ours = Z_ours[idx_ours, 2]
    h_sp   = Z_sp[idx_sp,   2]

    assert np.allclose(h_ours, h_sp, rtol=1e-6, atol=1e-8), f"height mismatch for method={method}"

@pytest.mark.parametrize("method", CONSTRAINT_METHODS)
def test_accepts_condensed_and_square_equivalently(method):
    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.normal(size=(8, 3))
    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    Z_sq = constrained_linkage(D, method=method)
    Z_cd = constrained_linkage(d_cond, method=method)

    for k in (2, 3, 4):
        a = hierarchy.fcluster(Z_sq, k, criterion="maxclust")
        b = hierarchy.fcluster(Z_cd, k, criterion="maxclust")
        assert partitions_match(a, b)

def test_invalid_method_raises():
    with pytest.raises(ValueError):
        constrained_linkage(np.array([[0.0, 1.0],[1.0, 0.0]]), method="not-a-method")

def test_all_zero_distances_produces_valid_linkage():
    # Three identical points
    X = np.zeros((3, 2))
    D = euclidean_square(X)
    Z = constrained_linkage(D, method="average")

    # Shape should be (n-1, 4)
    assert Z.shape == (2, 4)

    # Distances should be all zeros
    assert np.allclose(Z[:, 2], 0.0)

    # Clustering should succeed without error
    labels = hierarchy.fcluster(Z, 1, criterion="maxclust")
    assert len(np.unique(labels)) == 1