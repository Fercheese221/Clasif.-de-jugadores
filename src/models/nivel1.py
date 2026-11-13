"""Nivel 1: escalado -> Fisher -> KNN, con salida de probabilidades."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .fisher import FisherLDA
from .knn import KNNClassifier
from .metricas import f1_macro


class FisherKNN:
    def __init__(self, k=25):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.fisher_ = FisherLDA().fit(Xs, y)
        self.knn_ = KNNClassifier(k=self.k).fit(self.fisher_.transform(Xs), y)
        self.classes_ = self.knn_.classes_
        return self

    def transformar(self, X):
        return self.fisher_.transform(self.scaler_.transform(np.asarray(X, dtype=float)))

    def predict_proba(self, X):
        return self.knn_.predict_proba(self.transformar(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def curva_k_fisher(X, y, ks, n_splits=5, seed=42):
    """Macro-F1 medio por k. Escalado y Fisher se reajustan dentro de cada
    pliegue; los vecinos se calculan una sola vez con el k máximo."""
    X = np.asarray(X, dtype=float)
    clases, y_idx = np.unique(y, return_inverse=True)
    C = len(clases)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    filas = []
    for a, b in skf.split(X, y_idx):
        sc = StandardScaler().fit(X[a])
        fi = FisherLDA().fit(sc.transform(X[a]), y_idx[a])
        Za, Zb = fi.transform(sc.transform(X[a])), fi.transform(sc.transform(X[b]))
        m = KNNClassifier().fit(Za, y_idx[a])
        _, idx = m.kneighbors(Zb, k=max(ks))
        for k in ks:
            pred = m.predict_from_neighbors(idx[:, :k])
            filas.append({"k": k, "f1": f1_macro(y_idx[b], pred, C)})
    return pd.DataFrame(filas).groupby("k")["f1"].mean()
