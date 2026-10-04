from src.data_loader import get_iris_data
from src.model import train_kmeans, evaluate_clusters
from src.visualize import plot_clusters

def main():
    print("Initializing Iris Clustering Pipeline...\n")

    # Step 1: Data Loading
    X, y_true, feature_names, target_names, df = get_iris_data()
    print("Dataset Summary:")
    print(df.head())
    print("\n" + "="*50 + "\n")

    # Step 2: Model Training
    print("Training K-Means Model (k=3)...")
    kmeans = train_kmeans(X, n_clusters=3)
    y_kmeans = kmeans.labels_

    # Step 3: Model Evaluation
    acc, ari, cm_df = evaluate_clusters(y_true, y_kmeans, target_names)
    print(f"Cluster Alignment Accuracy: {acc * 100:.2f}%")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}\n")
    print("Confusion Matrix (True vs Matched Predicted Clusters):")
    print(cm_df)
    print("\n" + "="*50 + "\n")

    # Step 4: Visualization
    print("Generating Cluster Visualizations...")
    plot_clusters(X, y_kmeans, y_true, kmeans.cluster_centers_, target_names)

if __name__ == '__main__':
    main()