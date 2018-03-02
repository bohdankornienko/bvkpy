import numpy as np

def cm_score(cm):
    return np.mean(np.diag(cm) / np.sum(cm, axis=1))
