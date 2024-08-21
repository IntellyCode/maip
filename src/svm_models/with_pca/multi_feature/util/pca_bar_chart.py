import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca_bar_chart(X, feature_names):

    # Initialize PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)

    # Get PCA components (loadings)
    components = pca.components_

    # Get the absolute values of the components
    component_importance = np.abs(components)

    # Average the absolute importance for each original feature
    feature_importance = np.mean(component_importance, axis=0)

    # Sort feature importances in descending order
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_feature_names = np.array(feature_names)[sorted_indices]
    sorted_importance = feature_importance[sorted_indices]

    # Plot the bar graph
    plt.figure(figsize=(12, 8))
    plt.barh(sorted_feature_names, sorted_importance)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance from PCA')
    plt.grid(True)
    plt.show()
