import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def train_svm(features, labels):
    logger = logging.getLogger(__name__)
    X = np.array(features)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_model = SVC(kernel="rbf")
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info(f"Classification report: {classification_report(y_test, y_pred, target_names=le.classes_)}")