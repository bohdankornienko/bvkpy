import numpy as np
import itertools

from sklearn.metrics import confusion_matrix

def cm_score(cm):
    return np.mean(np.diag(cm) / np.sum(cm, axis=1))


class GridSearcher:
    def __init__(self, estimator, data):
        """
        @param data: dict with following fields: X_train, y_train, X_test, y_test
        """
        self._estimator = estimator
        self._data = data
        self._best_score = 0
        self._best_params = {}

    def search(self, opts):
        keys, values = zip(*opts.items())
        for v in itertools.product(*values):
            experiment = dict(zip(keys, v))

            self._estimator.set_params(**experiment)
            self._estimator.fit(self._data['X_train'], self._data['y_train'])

            y_pred = self._estimator.predict(self._data['X_test'])
            cm = confusion_matrix(self._data['y_test'], y_pred)

            score_cm = cm_score(cm)

            if score_cm > self._best_score:
                self._best_score = score_cm
                self._best_params = experiment

        return self._best_params, self._best_score

