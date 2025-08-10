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
  - **Min/max cluster size** penalties (linear in violation amount)
- No SciPy dependency — output `Z` works with SciPy’s downstream tools.

---

## 🔧 Install

```bash
# from source:
pip install "git+https://github.com/jonnevd/constrained-linkage"
```

---

## 🚀 Usage Example

```python
import numpy as np
from constrained_linkage import constrained_linkage
from scipy.cluster import hierarchy as hierarchy
from scipy.spatial.distance import squareform

# ==== Example 1: Using a constraint matrix ====

# 4 points in 1D space
X = np.array([[0.0], [0.1], [10.0], [10.1]])
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

# Constraint matrix: discourage merging points 0 and 1 (shouldnot-link)
M = np.zeros_like(D)
M[0, 1] = M[1, 0] = 1.0   # Positive values discourage merges
# Could also use negative values to encourage must-link merges

# Run constrained linkage
Z_con = constrained_linkage(
    D, method="average", 
    constraint_matrix=M, 
    normalize_distances=True
)

# Cluster into 2 groups
labels_con = hierarchy.fcluster(Z_con, 2, criterion="maxclust")
print("Cluster labels (with shouldnot-link constraint):", labels_con)


# ==== Example 2: Enforcing a maximum cluster size ====

# 6 points in 1D space (three tight pairs)
X = np.array([[0.0], [0.1], [5.0], [5.1], [10.0], [10.1]])
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))

# Run constrained linkage with max cluster size = 2
Z_max_size = constrained_linkage(
    D, method="average",
    max_cluster_size=2,
    max_penalty_weight=0.5,
    normalize_distances=True
)

# Cluster into 3 groups (will respect size limit)
labels_max = hierarchy.fcluster(Z_max_size, 3, criterion="maxclust")
print("Cluster labels (with max size = 2):", labels_max)
```