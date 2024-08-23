import logging
import os
import librosa.core as lc
import json as j
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC


class Trainer:
    GENRES = ["classical", "reggae", "rock"]
    PATH = "/Users/zeniosd/Documents/Programs/Python/maip/data/raw/"
    TXT_PATH = "/Users/zeniosd/Documents/Programs/Python/maip/data/objects/"

    def __init__(self):
        state = open(os.path.join(Trainer.TXT_PATH, "state.json"), "r")
        state_dict = j.load(state)

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s -  %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
            ]
        )
        if state_dict.key("genres") != Trainer.GENRES:
            state.close()
            return

        m = state_dict.key("music")
        l = state_dict.key("labels")
        self.music,self.labels = [], []
        if m is not None and l is not None:
            self.music, self.labels = m, l
        state.close()

        self.accuracy = 0
        self.classification = None

    def get_data(self):
        return self.music, self.labels

    def get_logger(self):
        return self.logger

    def load(self):
        for genre in Trainer.GENRES:
            path = os.path.join(Trainer.PATH + genre)
            for file in os.listdir(path):
                f_path = os.path.join(path, file)
                self.logger.info(f"Loading {genre} features for {f_path}")
                y, sr = lc.load(f_path)
                self.music.append((y, sr))
                self.labels.append(genre)

    def train_svm(self, features, C=1.0, kernel="rbf", gamma="scale"):
        X_train, X_test, y_train, y_test, _ = self._data_collection(features)

        svm_model = SVC(C=C, kernel=kernel, gamma=gamma)
        svm_model.fit(X_train, y_train)

        y_pred = svm_model.predict(X_test)

        self.accuracy = accuracy_score(y_test, y_pred)
        self.classification = classification_report(y_test, y_pred)

    def _data_collection(self,features):
        X = np.array(features)
        le = LabelEncoder()
        y = le.fit_transform(self.labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, le

    def grid_search(self, features, cv=10, scoring="accuracy",n_jobs=-1):
        X_train, X_test, y_train, y_test, le = self._data_collection(features)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly']
        }
        svm_model = SVC()
        grid_search = GridSearchCV(svm_model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        self.logger.info(f"Best Parameters: {grid_search.best_params_}")
        self.logger.info(f"Best Score: {grid_search.best_score_}")
        self.accuracy = accuracy_score(y_test, y_pred)
        self.classification = classification_report(y_test, y_pred)

    def __del__(self):
        state = open(os.path.join(Trainer.TXT_PATH, "state.json"), "w")
        dic = {
            "genres": Trainer.GENRES,
            "music": self.music,
            "labels": self.labels,
        }
        j.dump(dic, state)
        state.close()
