# 📊 Level 2 - Task 1 : Regression Analysis
**Internship** : Codveda Technologies — Data Analytics  
**Level** : 2 (Intermediate)  
**Author**: BATALONG MADDIE
**Dataset** : Boston Housing (506 observations, 13 features)

---

## 🎯 Objective
Predict the **median house value** (`MEDV`) using regression models,
and compare their performance to identify the best approach.

---

## 📁 Project Structure
```
/
└── task1_regression/
    ├── README.md
    ├── regression.py
    └── results/
        ├── correlation_matrix.png
        ├── r2_comparison.png
        ├── rmse_comparison.png
        ├── actual_vs_predicted.png
        ├── residuals_comparison.png
        └── rf_feature_importance.png
```

---

## 🛠️ Tools & Libraries
| Library | Usage |
|---------|-------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots |
| `scikit-learn` | ML models and evaluation |

---

## ⚙️ Models Trained
| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model |
| Ridge (α=1, 10) | L2 regularization |
| Lasso (α=0.1, 1) | L1 regularization |
| Random Forest | 100 trees, non-linear |

---

## 📈 Results

| Model              |    MSE  |  RMSE  |   R²   |
|--------------------|---------|--------|--------|
| Linear Regression  | 23.3351 | 4.8306 | 0.6818 |
| Ridge (alpha=1)    | 23.3462 | 4.8318 | 0.6816 |
| Ridge (alpha=10)   | 23.4776 | 4.8454 | 0.6799 |
| Lasso (alpha=0.1)  | 24.2129 | 4.9207 | 0.6698 |
| Lasso (alpha=1)    | 25.7306 | 5.0725 | 0.6491 |
| **Random Forest**  |  **8.6258** | **2.9370** | **0.8824** |

---

## 🔍 Key Findings

### Best Model : Random Forest
- R² = **0.88** → explains 88% of price variance
- RMSE = **2.94** → average error of ~$2,940 on house price

### Most Important Features (Random Forest)
| Rank | Feature | Importance | Meaning |
|------|---------|------------|---------|
| 1 | `RM` | 50.3% | Avg number of rooms per dwelling |
| 2 | `LSTAT` | 31.1% | % lower status of the population |
| 3 | `DIS` | 6.2% | Distance to employment centers |
| 4 | `CRIM` | 3.6% | Per capita crime rate |
| 13 | `CHAS` | 0.0% | Charles River proximity (no impact) |

### Observations
- `RM` and `LSTAT` together account for **81% of prediction power**
- `CHAS` (river proximity) has **zero importance** in the Random Forest
- Ridge and Lasso offer **no significant improvement** over Linear Regression
  on this dataset → multicollinearity is not a major issue here
- Random Forest **outperforms all linear models** by capturing
  non-linear relationships between features and price

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/Iris-sweety/Codveda-data-analytics-level2.git
cd codveda-data-analytics-level2/task1_regression
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Launch the script
```bash
 python regression.py
```

---

## 🔮 Possible Improvements
- Hyperparameter tuning with `GridSearchCV` on Random Forest
- Try `GradientBoostingRegressor` or `XGBoost`
- Apply cross-validation (k-fold) for more robust evaluation
- Feature engineering (interaction terms, polynomial features)

---

*Codveda Technologies Internship — Data Analytics | 2025*