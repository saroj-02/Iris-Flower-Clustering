import pandas as pd
from sklearn.datasets import load_iris

def get_iris_data():
    """
    Loads the Iris dataset from scikit-learn and returns a structured DataFrame
    along with feature data, target labels, and class names.
    """
    iris = load_iris()
    X = iris.data
    y_true = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    df = pd.DataFrame(X, columns=feature_names)
    df['True_Species'] = y_true

    return X, y_true, feature_names, target_names, df