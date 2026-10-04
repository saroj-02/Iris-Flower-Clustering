# 🪻 Iris Flower Clustering: An Unsupervised Machine Learning Pipeline

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

This repository presents an end-to-end **Unsupervised Machine Learning project** focused on partitioning the classic **Iris Flower Dataset** into distinct natural clusters using **K-Means Clustering ($k=3$)**. 

Because high-dimensional data cannot be rendered directly on 2D display media, this project incorporates **Principal Component Analysis (PCA)** to project the 4-dimensional feature space (`sepal length`, `sepal width`, `petal length`, and `petal width`) down to 2 principal components. Finally, we map the algorithm's unlabelled cluster outputs onto ground-truth species labels (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*) to rigorously evaluate clustering fidelity using quantitative metrics including **Alignment Accuracy**, **Adjusted Rand Index (ARI)**, and **Silhouette Analysis**.

---

## 🌸 Domain & Dataset Overview

The Iris dataset, introduced by British statistician and biologist Ronald Fisher in his 1936 paper *"The use of multiple measurements in taxonomic problems"*, remains a foundational benchmark in multivariate pattern recognition.

<Image src="image_agent_tag_8803831691368333215" alt="Three species of Iris flowers: Setosa, Versicolor, and Virginica with sepal and petal annotations" caption="Morphological comparison of Iris species" />

---

### Dataset Specifications

| Attribute | Value | Description |
| :--- | :--- | :--- |
| **Total Samples** | $150$ | 50 instances per class |
| **Feature Dimension** | $4$ | Continuous physical measurements in centimeters ($\text{cm}$) |
| **Target Classes** | $3$ | *Iris-setosa*, *Iris-versicolor*, *Iris-virginica* |
| **Missing Values** | $0$ | Complete dataset with no missing entries |
| **Class Distribution** | Balanced | $33.3\%$ per class ($50 / 50 / 50$) |

### Feature Attributes
1. **Sepal Length ($\text{cm}$)**: Length of the outer floral envelope.
2. **Sepal Width ($\text{cm}$)**: Width of the outer floral envelope.
3. **Petal Length ($\text{cm}$)**: Length of the inner floral petals.
4. **Petal Width ($\text{cm}$)**: Width of the inner floral petals.

---

## 📐 Mathematical Framework & Methodology

### 1. K-Means Clustering Algorithm

The K-Means algorithm partitions $N$ observations into $K$ disjoint clusters $S = \{S_1, S_2, \dots, S_K\}$, where each observation belongs to the cluster with the nearest mean (centroid). The objective function minimizes the **Within-Cluster Sum of Squares (WCSS)**:

$$\arg\min_{S} \sum_{i=1}^{K} \sum_{\mathbf{x} \in S_i} \Vert{}\mathbf{x} - \boldsymbol{\mu}_i\Vert{}^2$$

Where:
* $\mathbf{x} \in \mathbb{R}^4$ is a 4-dimensional feature vector.
* $\boldsymbol{\mu}_i \in \mathbb{R}^4$ represents the centroid vector of cluster $S_i$.
* $\Vert{}\mathbf{x} - \boldsymbol{\mu}_i\Vert{}^2$ is the squared Euclidean distance between data point $\mathbf{x}$ and centroid $\boldsymbol{\mu}_i$.

---

### 2. Dimensionality Reduction via PCA

To map 4D data into 2D Cartesian space without sacrificing variance structure, we compute the linear transformation matrix $\mathbf{W} \in \mathbb{R}^{4 \times 2}$ composed of eigenvectors corresponding to the top two eigenvalues of the empirical covariance matrix $\boldsymbol{\Sigma}$:

$$\boldsymbol{\Sigma} = \frac{1}{N-1} \mathbf{X}^T \mathbf{X}$$

$$\mathbf{X}_{\text{reduced}} = \mathbf{X} \cdot \mathbf{W}$$

The first two principal components ($PC_1, PC_2$) capture over **$95\%$ of the total cumulative variance** present in the raw features.

---

### 3. Cluster Alignment & Majority Matching

Because K-Means is strictly unsupervised, cluster assignment indices ($c \in \{0, 1, 2\}$) are arbitrary integer tokens with no inherent semantic link to target labels ($y \in \{0, 1, 2\}$). To evaluate performance against ground-truth labels:

$$\hat{y}(S_i) = \text{Mode}(\{y_j \mid \mathbf{x}_j \in S_i\})$$

Each cluster index $S_i$ is mapped to the modal ground-truth label among its enclosed points.

---

## 🏗️ Software Architecture & Directory Layout

The codebase follows modular design principles, isolating data loading, model fitting, metric evaluation, and visual rendering into independent modules under `src/`.

```text
iris_flower_clustering/
│
├── data/
│   └── iris.csv              # Optional raw dataset storage
│
├── src/
│   ├── __init__.py           # Package initialization marker
│   ├── data_loader.py        # Data ingestion and DataFrame schema setup
│   ├── model.py              # KMeans initialization, training & label mapping
│   └── visualize.py          # PCA computation & Seaborn scatter plotting
│
├── main.py                   # Main pipeline orchestrator script
├── requirements.txt          # Python package dependency list
├── .gitignore                # Byte-compiled & virtual environment exclusions
└── README.md                 # Complete project documentation