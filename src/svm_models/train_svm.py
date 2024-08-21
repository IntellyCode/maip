import logging
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def train_svm(features, labels):
    logger = logging.getLogger(__name__)

    # Convert features and labels to numpy arrays
    X = np.array(features)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }

    # Create an SVM model
    svm_model = SVC()

    # Initialize Grid Search with cross-validation
    grid_search = GridSearchCV(svm_model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)

    # Fit the Grid Search
    grid_search.fit(X_train, y_train)

    # Get the best model from Grid Search
    best_model = grid_search.best_estimator_

    # Predict using the best model
    y_pred = best_model.predict(X_test)

    # Log the results
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    logger.info(f"Best Score: {grid_search.best_score_}")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info(f"Classification Report: {classification_report(y_test, y_pred, target_names=le.classes_)}")

