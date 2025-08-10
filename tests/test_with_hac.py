import numpy as np
import pytest
sp = pytest.importorskip("scipy")  # skip entire module if SciPy isn't installed

from constrained_linkage import constrained_linkage
from scipy.cluster import hierarchy as hierarchy
from scipy.spatial.distance import squareform

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
    X, _ = make_sines(n_groups=3, per_group=5, random_state=1)
    D = euclidean_square(X)
    # Our linkage
    Z_ours = constrained_linkage(D, method="average", random_state=0)
    # SciPy linkage on the same distances
    d_cond = squareform(D, checks=False)
    Z_sp = hierarchy.linkage(d_cond, method="average", optimal_ordering=False)

    # Compare partitions at k=3 clusters
    k = 3
    labels_ours = hierarchy.fcluster(Z_ours, k, criterion="maxclust")
    labels_sp = hierarchy.fcluster(Z_sp, k, criterion="maxclust")
    assert partitions_match(labels_ours, labels_sp)

def test_constraints_push_within_group_merges_first():
    X, true_g = make_sines(n_groups=3, per_group=4, random_state=2)
    D = euclidean_square(X)
    n = D.shape[0]
    # Penalize cross-group merges heavily
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if true_g[i] != true_g[j]:
                M[i, j] = 1.0  # discourage cross-group merges

    Z_con = constrained_linkage(D, method="average", constraint_matrix=M, random_state=0)

    # The partition at k=3 should align with true groups
    labels_con = hierarchy.fcluster(Z_con, 3, criterion="maxclust")
    assert partitions_match(labels_con, true_g)

def test_max_cluster_size_changes_partition():
    # Simple dataset where three very close points would normally merge into one cluster
    X = np.array([[0.00], [0.01], [0.02],   # tight triplet
                  [10.0], [20.0], [30.0]])  # far-apart singles

    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    # Baseline: SciPy average linkage
    Z_sp = hierarchy.linkage(d_cond, method="average")
    labels_sp = hierarchy.fcluster(Z_sp, 3, criterion="maxclust")

    # Constrained: forbid clusters larger than 2
    Z_con = constrained_linkage(
        D,
        method="average",
        max_cluster_size=2,
        max_penalty_weight=0.2,
        normalize_distances=True,
        random_state=0,
    )
    labels_con = hierarchy.fcluster(Z_con, 3, criterion="maxclust")

    # They should differ — constraint stops the triplet merge
    assert not partitions_match(labels_sp, labels_con)

def test_constraint_matrix_should_and_shouldnot_link():
    # Four points in 1D space:
    # A and D are far apart -> should-link (negative penalty will pull them together)
    # A and B are very close -> shouldnot-link (positive penalty will push them apart)
    X = np.array([
        [0.0],   # 0: A
        [0.1],   # 1: B
        [5.0],   # 2: C
        [10.0],  # 3: D
    ])
    D = euclidean_square(X)
    n = D.shape[0]

    # Build constraint matrix
    M = np.zeros((n, n))

    # Must-link: A (0) and D (3) — strong negative penalty
    M[0, 3] = M[3, 0] = -0.5

    # Cannot-link: A (0) and B (1) — strong positive penalty
    M[0, 1] = M[1, 0] = 0.5

    # Run constrained linkage with normalized distances so penalties are comparable
    Z = constrained_linkage(
        D,
        method="average",
        constraint_matrix=M,
        random_state=0,
        normalize_distances=True
    )

    # Get 2-cluster partition
    labels = hierarchy.fcluster(Z, 2, criterion="maxclust")

    # Should-link check: A and D together
    assert labels[0] == labels[3]
    # Shoudlnot-link check: A and B apart
    assert labels[0] != labels[1]

def test_max_penalty_enforces_equal_sized_clusters():
    # Arrange: tight group of 4 and two far-away points
    X = np.array([
        [0.00], [0.01], [0.02], [0.03],  # tight group
        [10.0],                          # singleton
        [20.0],                          # singleton
    ])
    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    # Without constraint: tight group merges into one big cluster early
    Z_sp = hierarchy.linkage(d_cond, method="average")
    labels_sp = hierarchy.fcluster(Z_sp, 3, criterion="maxclust")
    sizes_sp = sorted(np.bincount(labels_sp)[1:])

    # With constraint: max cluster size 2 and penalty
    Z_con = constrained_linkage(
        D,
        method="average",
        max_cluster_size=2,
        max_penalty_weight=0.3,
        normalize_distances=True,
        random_state=0,
    )
    labels_con = hierarchy.fcluster(Z_con, 3, criterion="maxclust")
    sizes_con = sorted(np.bincount(labels_con)[1:])

    # Assert: constrained case yields all equal-sized clusters of size 2
    assert sizes_con == [2, 2, 2]

    # Assert: without constraints, the sizes are *not* all equal
    assert sizes_sp != sizes_con


def test_min_size_constraint_reduces_singletons():
    # dataset: three very close points + several spaced points,
    # so unconstrained average linkage tends to leave at least one singleton at k=4
    X = np.array([
        [0.00], [0.01], [0.02],       # tight triplet
        [5.0], [10.0], [15.0], [20.0], [25.0]  # spaced singles
    ])
    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    # Unconstrained baseline
    Z_sp = hierarchy.linkage(d_cond, method="average", optimal_ordering=False)
    labels_sp = hierarchy.fcluster(Z_sp, 4, criterion="maxclust")
    sizes_sp = sorted(np.bincount(labels_sp)[1:])  # drop label 0
    min_size_sp = min(sizes_sp)

    # Constrained: enforce a minimum size tendency.
    # Note: In our implementation, min penalty is an *additive* term applied when
    # the RESULTING merge size s < Cmin. Using a positive weight discourages
    # forming too-small merges; choose Cmin=3 to steer away from tiny clusters.
    Z_con = constrained_linkage(
        D,
        method="average",
        min_cluster_size=3,
        min_penalty_weight=0.8,   # penalize merges that would form size < 3
        normalize_distances=True, # put distances on [0,1] so 0.5 is meaningful
        random_state=0,
    )
    labels_con = hierarchy.fcluster(Z_con, 4, criterion="maxclust")
    sizes_con = sorted(np.bincount(labels_con)[1:])
    min_size_con = min(sizes_con)

    # The minimum cluster size with the constraint should be >= the unconstrained one,
    # and typically > 1 for this dataset.
    assert min_size_con >= min_size_sp
    assert min_size_con >= 2

def test_penalty_normalization_effect():
    # Points far apart so base distances are huge
    X = np.array([[0.0], [1000.0], [2000.0]])
    D = euclidean_square(X)

    # Very large scale distances without normalization → penalty negligible
    Z_no_norm = constrained_linkage(
        D,
        method="average",
        max_cluster_size=2,
        max_penalty_weight=0.5,  # Should be small relative to raw distances
        normalize_distances=False,
        random_state=0
    )

    # Same penalty, but with normalization → penalties now in same scale as distances
    Z_norm = constrained_linkage(
        D,
        method="average",
        max_cluster_size=2,
        max_penalty_weight=0.5,
        normalize_distances=True,
        random_state=0
    )

    # The merge distances should differ significantly
    assert not np.allclose(Z_no_norm[:, 2], Z_norm[:, 2])

@pytest.mark.parametrize("method", ["single", "complete", "average", "weighted", "centroid", "median", "ward"])
def test_method_parity_partitions_with_scipy(method):
    # deterministic medium-sized dataset
    rng = np.random.default_rng(123)
    X = rng.normal(size=(12, 4))

    # Euclidean square + condensed (SciPy expects condensed)
    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    # our linkage (no constraints)
    Z_ours = constrained_linkage(D, method=method, random_state=0)

    # SciPy linkage
    Z_sp = hierarchy.linkage(d_cond, method=method, optimal_ordering=False)

    # compare partitions at multiple k; partitions should be identical (up to relabeling)
    for k in (2, 3, 4, 5):
        labels_ours = hierarchy.fcluster(Z_ours, k, criterion="maxclust")
        labels_sp = hierarchy.fcluster(Z_sp, k, criterion="maxclust")
        assert partitions_match(labels_ours, labels_sp), f"partition mismatch for method={method}, k={k}"

@pytest.mark.parametrize("method", ["single", "complete", "average", "weighted", "centroid", "median", "ward"])
def test_method_parity_heights_close(method):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 3))
    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    Z_ours = constrained_linkage(D, method=method, random_state=0)
    Z_sp = hierarchy.linkage(d_cond, method=method, optimal_ordering=False)

    # sort by height then by pair (stable compare)
    idx_ours = np.lexsort((Z_ours[:,0], Z_ours[:,1], Z_ours[:,2]))
    idx_sp   = np.lexsort((Z_sp[:,0],   Z_sp[:,1],   Z_sp[:,2]))

    # heights only
    h_ours = Z_ours[idx_ours, 2]
    h_sp   = Z_sp[idx_sp,   2]

    assert np.allclose(h_ours, h_sp, rtol=1e-6, atol=1e-8), f"height mismatch for method={method}"

def test_accepts_condensed_and_square_equivalently():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(8, 3))
    D = euclidean_square(X)
    d_cond = squareform(D, checks=False)

    Z_sq = constrained_linkage(D, method="average", random_state=0)
    Z_cd = constrained_linkage(d_cond, method="average", random_state=0)

    # compare partitions at several k
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
    Z = constrained_linkage(D, method="average", random_state=0)

    # Shape should be (n-1, 4)
    assert Z.shape == (2, 4)

    # Distances should be all zeros
    assert np.allclose(Z[:, 2], 0.0)

    # Clustering should succeed without error
    labels = hierarchy.fcluster(Z, 1, criterion="maxclust")
    assert len(np.unique(labels)) == 1