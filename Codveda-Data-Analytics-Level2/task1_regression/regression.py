import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df=pd.read_csv('data\\cleaned_house_data.csv')
print(df.head())


# Separate features and target
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Train set : {X_train.shape}")
print(f"Test set  : {X_test.shape}")
print(" Features scaled successfully")

# Define all models
models = {
    "Linear Regression" : LinearRegression(),
    "Ridge (alpha=1)"   : Ridge(alpha=1.0),
    "Ridge (alpha=10)"  : Ridge(alpha=10.0),
    "Lasso (alpha=0.1)" : Lasso(alpha=0.1),
    "Lasso (alpha=1)"   : Lasso(alpha=1.0),
    "Random Forest"     : RandomForestRegressor(n_estimators=100, random_state=42),
}

# Train and evaluate each model
results = {}

print(f"{'Model':<25} {'MSE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 55)

for name, model in models.items():
    if name == "Random Forest":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        "MSE" : round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2"  : round(r2, 4),
        "preds": y_pred
    }

    print(f"{name:<25} {mse:>8.4f} {rmse:>8.4f} {r2:>8.4f}")

# Build results dataframe
results_df = pd.DataFrame({
    name: {"MSE": v["MSE"], "RMSE": v["RMSE"], "R²": v["R2"]}
    for name, v in results.items()
}).T.reset_index().rename(columns={"index": "Model"})

# R² comparison bar chart
plt.figure(figsize=(10, 5))
bars = plt.barh(results_df["Model"], results_df["R²"],
                color=["blue", "orange", "orange",
                       "green", "green", "red"])
plt.axvline(x=results_df["R²"].max(), color="black",
            linestyle="--", lw=1, label=f"Best R² = {results_df['R²'].max():.3f}")
plt.xlabel("R² Score")
plt.title("Model Comparison — R² Score ")
plt.legend()
plt.tight_layout()
plt.savefig("task1_regression\\results\\r2_comparison.png", dpi=150)
plt.show()

# RMSE comparison bar chart
plt.figure(figsize=(10, 5))
plt.barh(results_df["Model"], results_df["RMSE"],
         color=["blue", "orange", "orange",
                "green", "green", "red"])
plt.axvline(x=results_df["RMSE"].min(), color="black",
            linestyle="--", lw=1, label=f"Best RMSE = {results_df['RMSE'].min():.3f}")
plt.xlabel("RMSE")
plt.title("Model Comparison — RMSE ")
plt.legend()
plt.tight_layout()
plt.savefig("task1_regression\\results\\rmse_comparison.png", dpi=150)
plt.show()

# Scatter plot of actual vs predicted for the best model 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (name, vals) in enumerate(results.items()):
    r2 = vals["R2"]
    preds = vals["preds"]

    axes[i].scatter(y_test, preds, alpha=0.5, color="blue",
                    edgecolors="white", s=40)
    axes[i].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[i].set_title(f"{name}\nR² = {r2:.3f}")
    axes[i].set_xlabel("Actual Values")
    axes[i].set_ylabel("Predicted Values")

plt.suptitle("Actual vs Predicted — All Models", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("task1_regression\\results\\actual_vs_predicted.png", dpi=150)
plt.show()

#residual plot for the best model 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, name in zip(axes, ["Linear Regression", "Random Forest"]):
    residuals = y_test - results[name]["preds"]
    sns.histplot(residuals, bins=30, kde=True, ax=ax, color="green")
    ax.axvline(0, color="black", linestyle="--", lw=1.5)
    ax.set_title(f"Residuals — {name}")
    ax.set_xlabel("Residual (Actual - Predicted)")

plt.tight_layout()
plt.savefig("task1_regression\\results\\residuals_comparison.png", dpi=150)
plt.show()

# Extract Random Forest feature importance
rf_model = models["Random Forest"]

feat_imp = pd.DataFrame({
    "Feature"   : X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_imp, x="Importance", y="Feature")
plt.title("Random Forest — Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("task1_regression\\results\\rf_feature_importance.png", dpi=150)
plt.show()

print(feat_imp.to_string(index=False))

