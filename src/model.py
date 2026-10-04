import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix

def train_kmeans(X, n_clusters=3, random_state=42):
    """
    Trains a K-Means model with specified clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X)
    return kmeans

def evaluate_clusters(y_true, y_kmeans, target_names):
    """
    Aligns arbitrary K-Means cluster IDs with ground-truth species labels
    and computes clustering evaluation metrics.
    """
    # Map each cluster ID to the majority true class label
    labels = np.zeros_like(y_kmeans)
    for i in range(len(np.unique(y_kmeans))):
        mask = (y_kmeans == i)
        if np.sum(mask) > 0:
            labels[mask] = mode(y_true[mask], keepdims=True).mode[0]

    # Metrics
    acc = accuracy_score(y_true, labels)
    ari = adjusted_rand_score(y_true, y_kmeans)
    cm = confusion_matrix(y_true, labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

    return acc, ari, cm_df