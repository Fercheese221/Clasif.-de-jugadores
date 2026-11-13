"""Selección secuencial hacia adelante (SFS) con KNN como evaluador."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .knn import KNNClassifier
from .metricas import f1_macro, elegir_k


class ClaseUnica:
    """Submodelo trivial para bloques con una sola clase: probabilidad 1."""

    def __init__(self, clase):
        self.classes_ = np.array([clase])
        self.selected_ = []
        self.history_ = []
        self.k_ = None

    def predict_proba(self, X):
        return np.ones((len(X), 1))


class SFSKNN:
    """Escalado -> selección de métricas con SFS -> KNN.

    SFS agrega, en cada paso, la métrica que más mejora el macro-F1 medio en
    validación cruzada, y se detiene cuando la mejora es menor que `tol` o se
    alcanzan `max_features`. Después se elige k por validación cruzada sobre
    las métricas ya seleccionadas.
    """

    def __init__(self, k_sfs=15, ks=None, n_splits=3, max_features=15, tol=5e-4, seed=42, verbose=False):
        self.k_sfs = k_sfs
        self.ks = list(range(1, 52, 2)) if ks is None else list(ks)
        self.n_splits = n_splits
        self.max_features = max_features
        self.tol = tol
        self.seed = seed
        self.verbose = verbose

    def _puntaje(self, folds, cols, C):
        total = 0.0
        for Xa, ya, Xb, yb in folds:
            knn = KNNClassifier(k=self.k_sfs, batch_size=2000).fit(Xa[:, cols], ya)
            total += f1_macro(yb, knn.predict(Xb[:, cols]), C)
        return total / len(folds)

    def fit(self, X, y, nombres=None):
        X = np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        C = len(self.classes_)
        d = X.shape[1]
        self.nombres_ = list(nombres) if nombres is not None else [f"x{i}" for i in range(d)]

        n_splits = min(self.n_splits, int(np.bincount(y_idx).min()))
        if n_splits < 2:
            raise ValueError("Alguna clase tiene menos de 2 ejemplos; no se puede validar.")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        # El escalado se ajusta dentro de cada pliegue y se calcula una sola vez
        folds = []
        for a, b in skf.split(X, y_idx):
            sc = StandardScaler().fit(X[a])
            folds.append((sc.transform(X[a]), y_idx[a], sc.transform(X[b]), y_idx[b]))

        seleccionadas, restantes, historial, mejor = [], list(range(d)), [], -np.inf
        while restantes and len(seleccionadas) < self.max_features:
            puntajes = {j: self._puntaje(folds, seleccionadas + [j], C) for j in restantes}
            j_mejor = max(puntajes, key=puntajes.get)
            if historial and puntajes[j_mejor] - mejor < self.tol:
                break
            seleccionadas.append(j_mejor)
            restantes.remove(j_mejor)
            mejor = puntajes[j_mejor]
            historial.append(mejor)
            if self.verbose:
                print(f"    + {self.nombres_[j_mejor]:<30s} macro-F1 CV = {mejor:.4f}")

        self.selected_ = seleccionadas
        self.history_ = historial

        # Elegir k con las métricas ya seleccionadas
        kmax = min(max(self.ks), min(len(f[1]) for f in folds) - 1)
        ks = [k for k in self.ks if k <= kmax]
        filas = []
        for Xa, ya, Xb, yb in folds:
            m = KNNClassifier().fit(Xa[:, seleccionadas], ya)
            _, idx = m.kneighbors(Xb[:, seleccionadas], k=kmax)
            for k in ks:
                filas.append({"k": k, "f1": f1_macro(yb, m.predict_from_neighbors(idx[:, :k]), C)})
        self.cv_k_ = pd.DataFrame(filas).groupby("k")["f1"].mean()
        self.k_ = elegir_k(self.cv_k_)

        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.knn_ = KNNClassifier(k=self.k_).fit(Xs[:, seleccionadas], y)
        return self

    @property
    def metricas_seleccionadas_(self):
        return [self.nombres_[j] for j in self.selected_]

    def predict_proba(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.knn_.predict_proba(Xs[:, self.selected_])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
