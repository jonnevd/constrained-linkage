[![Tests](https://github.com/jonnevd/constrained-linkage/actions/workflows/test.yml/badge.svg)](https://github.com/jonnevd/constrained-linkage/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/constrained-linkage.svg)](https://pypi.org/project/constrained-linkage/)

# Constrained Hierarchical Agglomerative Clustering
This repository contains the implementation of the constrained linkage function for Constrained Hierarchical Agglomerative Clustering from the paper:

> **HEAT: Hierarchical-constrained Encoder-Assisted Time series clustering for fault detection in district heating substations**  
> *Jonne van Dreven, Abbas Cheddad, Ahmad Nauman Ghazi, Sadi Alawadi, Jad Al Koussa, Dirk Vanhoudt*  
> *Energy and AI, 21 (2025), 100548*  
> DOI: [10.1016/j.egyai.2025.100548](https://doi.org/10.1016/j.egyai.2025.100548)

If you use this library in academic or scientific work, please cite:

```bibtex
@article{van_Dreven-HEAT,
  title={HEAT: Hierarchical-constrained Encoder-Assisted Time series clustering for fault detection in district heating substations},
  volume={21},
  ISSN={2666-5468},
  DOI={10.1016/j.egyai.2025.100548},
  journal={Energy and AI},
  author={van Dreven, Jonne and Cheddad, Abbas and Ghazi, Ahmad Nauman and Alawadi, Sadi and Al Koussa, Jad and Vanhoudt, Dirk},
  year={2025},
  month=sep,
  pages={100548}
}
```

A **NumPy-only** hierarchical agglomerative clustering routine with **soft constraints**, returning a SciPy-compatible linkage matrix `Z`.

## ✨ Features

- Drop-in replacement for a constrained `linkage` routine supporting:
  - `single`, `complete`, `average`, `weighted`, `centroid`, `median`, `ward`
- Accepts **either**:
  - condensed 1-D distances (`len n*(n-1)/2`)
  - `n×n` square distance matrix
- Adds **soft constraints**:
  - **Must-link / Cannot-link** via a constraint matrix `M`
    - `M[i,j] < 0` → encourages merging (must-link)
    - `M[i,j] > 0` → discourages merging (cannot-link)
    - When `normalize_distances=True`, these penalties are scaled relative to the [0, 1] normalized distance range, making them proportional regardless of the original distance scale.
  - **Min/max cluster size** penalties (linear in violation amount)
    - Similarly scales proportionally when `normalize_distances=True`
- No SciPy dependency — output `Z` works with SciPy’s downstream tools.

---

## 🔌 Plug-and-play

`constrained_linkage` is a **drop-in replacement** for SciPy’s `linkage` function.  

- **No constraints?** Works identically to `scipy.cluster.hierarchy.linkage`.  
- **With constraints?** Adds powerful, flexible soft constraints with minimal code changes.  
- Output is a **SciPy-compatible linkage matrix `Z`**, so you can keep using all SciPy tools (e.g., `fcluster`, `dendrogram`) unchanged.

---

## 🔧 Install

```bash
pip install constrained-linkage
# from source:
pip install "git+https://github.com/jonnevd/constrained-linkage"
```

---

## 🚀 Usage Examples

Below we illustrate **must-link** (negative penalties) and **cannot-link** (positive penalties) via the constraint matrix `M`.  
All distances are optionally scaled to `[0,1]` when `normalize_distances=True`, so penalties are **scale-free**.

> **Semantics:**  
> - `M[i, j] < 0` → **must-link** (encourage merging i↔j)  
> - `M[i, j] > 0` → **cannot-link** (discourage merging i↔j)

---

### Example 1 — Must-link & Cannot-link constraints

```python
import numpy as np
from constrained_linkage import constrained_linkage
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

# Four points in 1D (two well-separated pairs)
X = np.array([[0.0], [0.1], [10.0], [10.1]])
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

# Constraint matrix: must-link 0↔1, cannot-link 2↔3
M = np.zeros_like(D)
M[0, 1] = M[1, 0] = -0.6   # must-link (negative)
M[2, 3] = M[3, 2] =  0.6   # cannot-link (positive)

Z = constrained_linkage(
    D, method="average",
    constraint_matrix=M,
    normalize_distances=True
)

# Works seamlessly with SciPy tools
labels = hierarchy.fcluster(Z, 2, criterion="maxclust")
print("Partition with must-link(0,1) & cannot-link(2,3):", labels)

plt.figure(figsize=(6, 3))
hierarchy.dendrogram(Z, labels=[f"P{i}" for i in range(len(X))])
plt.title("Dendrogram — must-link(0,1), cannot-link(2,3)")
plt.tight_layout()
plt.show()
```

### Example 2 — Enforcing a maximum cluster size

Discourage clusters larger than a threshold by adding a positive penalty above the maximum.

```python
import numpy as np
from constrained_linkage import constrained_linkage
from scipy.cluster import hierarchy

# Six points in 1D (three tight pairs)
X = np.array([[0.0], [0.1], [5.0], [5.1], [10.0], [10.1]])
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

Z_max = constrained_linkage(
    D, method="average",
    max_cluster_size=2,     # soft cap
    max_penalty_weight=0.6, # stronger => avoids overgrown clusters
    normalize_distances=True
)

labels_max = hierarchy.fcluster(Z_max, 3, criterion="maxclust")
print("Partition with max_cluster_size=2:", labels_max)
```


### Example 3 — Enforcing a minimum cluster size

When domain knowledge suggests small units should coalesce before analysis, use a minimum size prior to avoid singletons or small groups. Increasing the penalty weight strengthens this bias, as shown in the figure below.

<p align="center">
  <img src="docs/min_cluster_effect.png" alt="Effect of min_cluster_size penalty on small clusters" width="500">
</p>

```python
import numpy as np
from constrained_linkage import constrained_linkage
from scipy.cluster import hierarchy

# Six points in 1D (three tight pairs)
X = np.array([[0.0], [0.1], [5.0], [5.1], [10.0], [10.1]])
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

Z_min = constrained_linkage(
    D, method="average",
    min_cluster_size=3,     # target minimum size
    min_penalty_weight=0.5, # stronger => merge undersized clusters earlier
    normalize_distances=True
)

labels_min = hierarchy.fcluster(Z_min, 2, criterion="maxclust")
print("Partition with min_cluster_size=3:", labels_min)
```