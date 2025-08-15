import numpy as np
import pytest
from constrained_linkage import constrained_linkage

VALID_METHODS = [
    "single", "complete", "average", "weighted", "centroid", "median", "ward"
]

def euclidean_square(X: np.ndarray) -> np.ndarray:
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

def test_shapes_and_monotonicity_single():
    X = np.array([[0.0], [0.1], [10.0], [10.1], [20.0]])
    D = euclidean_square(X)
    Z = constrained_linkage(D, method="single")
    n = D.shape[0]
    assert Z.shape == (n - 1, 4)
    # ids are within expected range
    ids = Z[:, :2].astype(int).ravel()
    assert ids.min() >= 0 and ids.max() <= 2*n - 2
    # sizes increase correctly
    assert Z[-1, 3] == n
    # distances are non-negative and non-decreasing for single linkage
    d = Z[:, 2]
    assert np.all(d >= -1e-12)
    assert np.all(d[1:] + 1e-12 >= d[:-1])

def test_constraints_do_not_break_output():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    D = euclidean_square(X)
    M = np.zeros_like(D)
    M[0,1] = M[1,0] = 5.0
    Z = constrained_linkage(
        D, method="average",
        constraint_matrix=M,
        min_cluster_size=3, max_cluster_size=3,
        min_penalty_weight=0.5, max_penalty_weight=0.25
    )
    n = D.shape[0]
    assert Z.shape == (n - 1, 4)
    assert np.all(Z[:, 2] >= 0.0)
    assert int(Z[-1, 3]) == n

def test_asymmetric_constraint_matrix_handling():
    X = np.array([[0.0], [1.0], [2.0]])
    D = euclidean_square(X)

    # Deliberately asymmetric matrix
    M = np.zeros_like(D)
    M[0, 1] = 1.0
    M[1, 0] = 0.0  # asymmetry

    try:
        Z = constrained_linkage(
            D,
            method="average",
            constraint_matrix=M
        )
        # If no error, ensure result is same as with symmetrized version
        M_sym = (M + M.T) / 2
        Z_sym = constrained_linkage(
            D,
            method="average",
            constraint_matrix=M_sym
        )
        assert np.allclose(Z, Z_sym)
    except ValueError:
        # If strict validation is implemented, asymmetric matrix should raise
        pass

def test_small_n_edge_cases():
    # Case n=1 — should raise (cannot cluster)
    with pytest.raises(ValueError):
        constrained_linkage(np.zeros((1, 1)), method="average")

    # Case n=2 — should return single merge
    D = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    Z = constrained_linkage(D, method="single")
    assert Z.shape == (1, 4)
    assert int(Z[0, 0]) == 0
    assert int(Z[0, 1]) == 1
    assert Z[0, 3] == 2

def test_constraint_matrix_not_numpy_array():
    with pytest.raises(TypeError, match="NumPy ndarray"):
        constrained_linkage(np.eye(3), method="single", constraint_matrix=[[0, 1], [1, 0]])

def test_constraint_matrix_wrong_shape():
    with pytest.raises(ValueError, match="must be shape"):
        constrained_linkage(np.eye(3), method="single", constraint_matrix=np.zeros((2, 2)))

def test_constraint_matrix_non_numeric_dtype():
    bad_matrix = np.array([["a", "b"], ["c", "d"]], dtype=object)
    with pytest.raises(TypeError, match="numeric dtype"):
        constrained_linkage(np.eye(2), method="single", constraint_matrix=bad_matrix)

def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        constrained_linkage(np.eye(2), method="not-a-method")

def test_min_cluster_size_too_small():
    with pytest.raises(ValueError, match="min_cluster_size must be >= 1"):
        constrained_linkage(np.eye(2), method=VALID_METHODS[0], min_cluster_size=0)

def test_max_cluster_size_too_small():
    with pytest.raises(ValueError, match="max_cluster_size must be >= 1"):
        constrained_linkage(np.eye(2), method=VALID_METHODS[0], max_cluster_size=0)

def test_min_greater_than_max_cluster_size():
    with pytest.raises(ValueError, match="cannot be greater"):
        constrained_linkage(np.eye(2), method=VALID_METHODS[0],
                            min_cluster_size=5, max_cluster_size=3)

def test_negative_penalty_weight():
    with pytest.raises(ValueError, match="must be non-negative"):
        constrained_linkage(np.eye(2), method=VALID_METHODS[0], min_penalty_weight=-0.1)
    with pytest.raises(ValueError, match="must be non-negative"):
        constrained_linkage(np.eye(2), method=VALID_METHODS[0], max_penalty_weight=-0.1)

def test_all_valid_passes():
    # This should NOT raise
    constrained_linkage(
        np.eye(3), method=VALID_METHODS[0],
        min_cluster_size=1, max_cluster_size=3,
        min_penalty_weight=0.5, max_penalty_weight=0.5,
        normalize_distances=True
    )