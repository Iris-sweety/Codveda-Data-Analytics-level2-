# Level 2 - Task 3 : Clustering Analysis (K-Means)
**Internship** : Codveda Technologies — Data Analytics  
**Level** : 2   
**Dataset** : Iris 

---

##  Objective
Group similar Iris flowers using **K-Means clustering** without using
the species labels, and evaluate how well the algorithm recovers
the natural groups.

---

##  Project Structure
```
level2/
└── task3_clustering/
    ├── README.md
    ├── clustering.py
    └── results/
        ├── pairplot.png
        ├── cluster_by_features.png
        ├── kmeans_pca.png
        ├── kmeans_evaluation.png
        ├── confusion_table.png
        └── cluster_boxplots.png
```

---

## 🛠️ Tools & Libraries
| Library | Usage |
|---------|-------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots |
| `scikit-learn` | Scaling, KMeans, PCA, metrics |

---

## ⚙️ Methodology
1. Load and explore the Iris dataset
2. Standardize features with `StandardScaler`
3. Find optimal k with Elbow Method + Silhouette Score
4. Train K-Means with k=3
5. Visualize clusters (PCA 2D + original features)
6. Compare clusters vs true species labels

---

## 📈 Results

### Optimal Number of Clusters

#### Elbow Method
The inertia drops sharply from k=2 (225) to k=3 (142),
then the curve flattens significantly.
The "elbow" is clearly visible at **k=3** → confirmed optimal choice.

| k | Inertia |
|---|---------|
| 2 | 225 |
| **3** | **142** ← elbow |
| 4 | 114 |
| 5 | 90 |
| 10 | 51 |

#### Silhouette Score
The highest score is at **k=2 (0.58)**, but k=2 would merge
Versicolor and Virginica into one group — biologically incorrect.
At **k=3 (0.46)** the score remains strong and matches the known
structure of the data.

| k | Silhouette Score |
|---|-----------------|
| 2 | 0.58 |
| **3** | **0.46** ← chosen |
| 4 | 0.39 |
| 5 | 0.35 |

> ⚠️ The silhouette favors k=2 mathematically, but domain knowledge
> confirms k=3 as the correct choice — there are 3 known Iris species.
> This is a good example of why metrics alone are not sufficient.

---

### K-Means Final Model (k=3)

| Metric | Value |
|--------|-------|
| Silhouette Score | 0.46 |
| Cluster 0 size | 53 flowers |
| Cluster 1 size | 50 flowers |
| Cluster 2 size | 47 flowers |

---

## 🔍 Visual Interpretations

### PCA 2D Projection
PC1 explains **72.8%** of variance and PC2 explains **23.0%**,
giving a total of **95.8% variance retained** in 2D — the
visualization is highly faithful to the original 4D data.

**What we see :**
- **Cluster 1 (green)** → perfectly isolated on the left (PC1 < -1.5)
  → corresponds to Setosa, very distinct from the others
- **Cluster 2 (blue)** → well separated on the right (PC1 > 1)
  → corresponds mostly to Virginica
- **Cluster 0 (red)** → center zone, partially overlapping with Cluster 2
  → corresponds mostly to Versicolor
- The **centroids (X)** are well centered within each group

---

### Sepal vs Petal Scatter Plots

**Sepal features (left plot) :**
- The 3 clusters overlap significantly on sepal dimensions
- Cluster 1 (Setosa) is slightly separated but not cleanly
- Sepal features alone are **poor discriminators**

**Petal features (right plot) :**
- Cluster 1 (Setosa) is **perfectly isolated** at the bottom left
  (petal length 1-2 cm, petal width 0-0.5 cm)
- Cluster 0 and Cluster 2 are well separated with minor overlap
- Petal features are the **best discriminators** for Iris species

---

### Confusion Table Analysis
```
True Species →    setosa   versicolor   virginica
Cluster 0              0           39          14
Cluster 1             50            0           0
Cluster 2              0           11          36
```

**Cluster 1 = Setosa** → **100% accuracy**
All 50 Setosa flowers are perfectly grouped. Setosa is so
distinct that K-Means isolates it without any error.

**Cluster 2 ≈ Virginica** → **76% accuracy** (36/47)
36 Virginica correctly grouped, but 11 Versicolor were
misclassified into this cluster due to feature overlap.

**Cluster 0 ≈ Versicolor** → **74% accuracy** (39/53)
39 Versicolor correctly grouped, but 14 Virginica were
pulled into this cluster — the two species share similar
petal dimensions in the boundary zone.

**Overall accuracy : ~83%** (125/150 correct assignments)

---

### Feature Distribution per Cluster (Boxplots)

**sepal_length :**
- Cluster 1 (Setosa) : 4.5 - 5.5 cm → smallest
- Cluster 0 (Versicolor) : 5.0 - 6.5 cm → medium
- Cluster 2 (Virginica) : 6.0 - 7.5 cm → largest
- Good separation between clusters

**sepal_width :**
- Cluster 1 (Setosa) : 3.0 - 4.2 cm → widest sepals
- Cluster 0 (Versicolor) : 2.2 - 3.2 cm → narrowest
- Cluster 2 (Virginica) : 2.5 - 3.5 cm → medium
- Setosa stands out clearly, but Versicolor and
  Virginica overlap → explains some misclassifications

**petal_length :**
- Cluster 1 (Setosa) : 1.0 - 2.0 cm → very short
- Cluster 0 (Versicolor) : 3.0 - 5.5 cm → medium
- Cluster 2 (Virginica) : 4.5 - 7.0 cm → long
- **Best feature** — near-perfect separation across clusters

**petal_width :**
- Cluster 1 (Setosa) : 0.1 - 0.5 cm → very thin
- Cluster 0 (Versicolor) : 1.0 - 2.0 cm → medium
- Cluster 2 (Virginica) : 1.5 - 2.5 cm → widest
- **Second best feature** — strong separation, minor overlap
  between Cluster 0 and Cluster 2

---

## ✅ Conclusion

| Metric | Value |
|--------|-------|
| Optimal k | 3 |
| Silhouette Score (k=3) | 0.46 |
| PCA variance retained | 95.8% |
| Setosa detection | **100%** (50/50) |
| Versicolor detection | **74%** (39/50) |
| Virginica detection | **76%** (36/50) |
| **Overall accuracy** | **~83%** (125/150) |

**Key findings :**
- K-Means successfully identifies 3 natural groups
  matching the known Iris species structure
- `petal_length` and `petal_width` are the strongest
  discriminating features — clearly visible in boxplots
  and scatter plots
- **Setosa is perfectly clustered** (100%) — it is
  biologically the most distinct species
- **Versicolor and Virginica partially overlap** — their
  boundary is not linearly separable, causing ~25%
  misclassification between the two
- The silhouette score of **0.46** indicates a good but
  not perfect clustering — expected given the natural
  overlap between Versicolor and Virginica

**Why the overlap between Versicolor and Virginica ?**
These two species evolved more recently from a common
ancestor, making them more similar morphologically than
either is to Setosa. K-Means (linear boundaries) cannot
perfectly separate them — a non-linear method would
perform better.

**Next steps :**
- Try **DBSCAN** for density-based clustering
- Try **Hierarchical Clustering** and compare dendrograms
- Use **t-SNE** instead of PCA for non-linear 2D projection
- Apply a classification model (Random Forest) to compare
  with supervised learning performance

---

*Codveda Technologies Internship — Data Analytics | 2026*