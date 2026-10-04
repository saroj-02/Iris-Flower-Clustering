import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

def plot_clusters(X, y_kmeans, y_true, cluster_centers, target_names):
    """
    Reduces feature dimensions using PCA to 2D and generates side-by-side
    scatter plots comparing K-Means predicted clusters vs. Ground Truth labels.
    """
    # PCA Dimensionality Reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(cluster_centers)

    df_plot = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    df_plot['Predicted_Cluster'] = [f'Cluster {c}' for c in y_kmeans]
    df_plot['True_Species'] = [target_names[i] for i in y_true]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: K-Means Predicted Clusters
    sns.scatterplot(
        data=df_plot, x='PCA1', y='PCA2', hue='Predicted_Cluster',
        palette='Set1', style='Predicted_Cluster', s=80, ax=axes[0]
    )
    axes[0].scatter(
        centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X',
        s=200, label='Centroids'
    )
    axes[0].set_title("K-Means Clustering (k=3)")
    axes[0].legend()

    # Plot 2: Ground Truth Species
    sns.scatterplot(
        data=df_plot, x='PCA1', y='PCA2', hue='True_Species',
        palette='Set1', style='True_Species', s=80, ax=axes[1]
    )
    axes[1].set_title("True Species Labels")

    plt.tight_layout()
    plt.show()