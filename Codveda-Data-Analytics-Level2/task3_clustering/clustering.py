import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df=pd.read_csv("data\\1) iris.csv")
print("Shape :", df.shape)
print("\nColumns :", df.columns.tolist())
print(df['species'].value_counts())
print(df.head())

features_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features_names]

print("Features used before scaling :")
print(X.columns.tolist())
print(X.describe().round(2))

#pairplot by true species
sns.pairplot(df[features_names + ['species']], hue='species', diag_kind='kde')
plt.suptitle("Iris — Pairplot by True Species", y=1.02, fontsize=13)
plt.savefig("task3_clustering\\results\\pairplot.png", dpi=150, bbox_inches="tight")
plt.show()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features_names)
print("\nFeatures after scaling :")
print(X_scaled_df.describe().round(2))

# Elbow method to find optimal k
inertia = []
silhouettes=[]
k_range=range(2, 11)

for k in k_range:
    km=KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))
    print(f"k={k} : Inertia={km.inertia_:.2f}, Silhouette Score={silhouette_score(X_scaled, km.labels_):.4f}")

fig,axes=plt.subplots(1,2, figsize=(12,5))
# Elbow plot
axes[0].plot(k_range, inertia, marker='o')
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].grid()
axes[0].axvline(x=3, color='red', linestyle='--', label='Optimal k=3')
axes[0].legend()

# Silhouette plot
axes[1].plot(k_range, silhouettes, marker='o', color='orange')
axes[1].set_title("Silhouette Scores")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid()
axes[1].axvline(x=3, color='red', linestyle='--', label='Optimal k=3')
axes[1].legend()

plt.suptitle("KMeans Clustering Evaluation", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("task3_clustering\\results\\kmeans_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

# Fit KMeans with optimal k
kmeans=KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster']=kmeans.fit_predict(X_scaled)
print("\nCluster distribution :")
print(df['cluster'].value_counts().sort_index())

print(f"\nFinal Silhouette Score for k=3 : {silhouette_score(X_scaled, df['cluster']):.4f}")
print(f"Final Inertia for k=3 : {kmeans.inertia_:.4f}")

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

#Project clusters on PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)

print(f"Variance explained by PC1 : {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Variance explained by PC2 : {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Total variance retained : {sum(pca.explained_variance_ratio_)*100:.2f}%")

palette = {0: "#E74C3C", 1: "#2ECC71", 2: "#3498DB"}

plt.figure(figsize=(9, 6))

for cluster_id in range(3):
    mask = df["cluster"] == cluster_id
    plt.scatter(
        df.loc[mask, "PC1"],
        df.loc[mask, "PC2"],
        label=f"Cluster {cluster_id}",
        color=palette[cluster_id],
        alpha=0.7, s=70, edgecolors="white", lw=0.5
    )

# Plot centroids
plt.scatter(
    centers_pca[:, 0], centers_pca[:, 1],
    c="black", marker="X", s=250,
    zorder=5, label="Centroids"
)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title("K-Means Clustering — PCA 2D Projection")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("task3_clustering\\results\\kmeans_pca.png", dpi=150, bbox_inches="tight")
plt.show()

#visualize clusters by true species

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#plot1 sepal features
for cluster_id in range(3):
    mask = df["cluster"] == cluster_id
    axes[0].scatter(
        df.loc[mask, "sepal_length"],
        df.loc[mask, "sepal_width"],
        label=f"Cluster {cluster_id}",
        color=palette[cluster_id],
        alpha=0.7, s=70, edgecolors="white", lw=0.5
    )
#plot2 petal features
for cluster_id in range(3):
    mask = df["cluster"] == cluster_id
    axes[1].scatter(
        df.loc[mask, "petal_length"],
        df.loc[mask, "petal_width"],
        label=f"Cluster {cluster_id}",
        color=palette[cluster_id],
        alpha=0.7, s=70, edgecolors="white", lw=0.5
    )

axes[0].set_xlabel("Sepal Length")
axes[0].set_ylabel("Sepal Width")
axes[0].set_title("Clusters by Sepal Features")
axes[0].legend()

axes[1].set_xlabel("Petal Length")
axes[1].set_ylabel("Petal Width")
axes[1].set_title("Clusters by Petal Features")
axes[1].legend()

plt.tight_layout()
plt.savefig("task3_clustering\\results\\clusters_by_features.png", dpi=150, bbox_inches="tight")
plt.show()

#clusters vs true species

#confusion table:clusters vs true species
confusion_table = pd.crosstab(
    df['cluster'],
    df['species'],
    rownames=['cluster'],
    colnames=['True Species']    
)
print("\nConfusion Table (Clusters vs True Species) :")
print(confusion_table)

# Visualize confusion table as heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(confusion_table, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Table: Clusters vs True Species")
plt.tight_layout()
plt.savefig("task3_clustering\\results\\confusion_table.png", dpi=150, bbox_inches="tight")
plt.show()

# Mean of each feature per cluster
print("Feature means per cluster :")
cluster_stats = df.groupby("cluster")[features_names].mean().round(3)
print(cluster_stats)

# Visualize with boxplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(features_names):
    for cluster_id in range(3):
        data = df[df["cluster"] == cluster_id][feature]
        axes[i].boxplot(data, positions=[cluster_id],
                        widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=list(palette.values())[cluster_id],
                                      alpha=0.7))
    axes[i].set_title(feature)
    axes[i].set_xlabel("Cluster")
    axes[i].set_ylabel("Value (cm)")
    axes[i].set_xticks([0, 1, 2])
    axes[i].grid(True, alpha=0.3)

plt.suptitle("Feature Distribution per Cluster", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("task3_clustering\\results\\cluster_boxplots.png", dpi=150)
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feature in enumerate(features_names):   
    for cluster_id in range(3):
        data = df[df["cluster"] == cluster_id][feature]
        axes[i].boxplot(data, positions=[cluster_id],
                        widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=list(palette.values())[cluster_id],
                                      alpha=0.7))
    axes[i].set_title(feature)
    axes[i].set_xlabel("Cluster")
    axes[i].set_ylabel("Value (cm)")
    axes[i].set_xticks([0, 1, 2])
    axes[i].grid(True, alpha=0.3)

plt.suptitle("Feature Distribution per Cluster", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("task3_clustering\\results\\cluster_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()