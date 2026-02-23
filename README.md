# 📊 Level 2 - Intermediate Data Analytics
**Internship** : Codveda Technology — Data Analytics  

---

## 📁 Project Structure
```
level2/
├── README.md
├── task1_regression/
│   ├── README.md
│   ├── regression_analysis.ipynb
│   └── outputs/
│       ├── correlation_matrix.png
│       ├── r2_comparison.png
│       ├── rmse_comparison.png
│       ├── actual_vs_predicted.png
│       ├── residuals_comparison.png
│       └── rf_feature_importance.png
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

## 📌 Tasks Overview

| Task | Topic | Dataset | Best Result |
|------|-------|---------|-------------|
| Task 1 | Regression Analysis | Boston Housing | R² = 0.88 (Random Forest) |
| Task 3 | Clustering (K-Means) | Iris | 83% accuracy (k=3) |

---

## 🛠️ Tools & Libraries
| Library | Usage |
|---------|-------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots |
| `scikit-learn` | ML models, scaling, PCA, metrics |


---

## 📋 Key Findings

### Task 1 — Regression Analysis
- **Random Forest** is the best model with **R² = 0.88**
- `RM` (rooms) and `LSTAT` (population status) drive **81%**
  of house price prediction
- Linear models (Ridge, Lasso) plateau around R² = 0.68


### Task 3 — Clustering (K-Means)
- Optimal **k=3** confirmed by both Elbow and Silhouette methods
- **Setosa perfectly isolated** (100% accuracy)
- Versicolor & Virginica overlap → 83% overall accuracy
- `petal_length` and `petal_width` are the strongest features
- PCA retains **95.8% variance** in 2D projection

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/Iris-sweety/Codveda-Data-Analytics-level2-.git
cd Codveda-Data-Analytics-Level2
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

3. Launch any python script
```bash
python task1_regression/regression.py
python task3_clustering/clustering.py
```

---

*Codveda Technologies Internship — Data Analytics | 2026*
