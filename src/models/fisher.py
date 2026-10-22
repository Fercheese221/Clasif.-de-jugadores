"""Análisis Discriminante de Fisher multiclase, implementado con NumPy/SciPy.

Busca las direcciones W que maximizan la separación entre clases relativa a
la dispersión dentro de cada clase:  J(W) = |W' Sb W| / |W' Sw W|

La solución son los autovectores del problema generalizado  Sb v = λ Sw v.
Con C clases hay a lo sumo C - 1 direcciones con λ > 0.
"""
import numpy as np
from scipy.linalg import eigh


class FisherLDA:
    def __init__(self, n_components=None, reg=1e-6):
        """
        n_components : int | None
            Dimensiones de salida (máx. C - 1). None usa C - 1.
        reg : float
            Regularización relativa añadida a la diagonal de Sw. Las métricas
            muy correlacionadas vuelven Sw casi singular; esto la estabiliza.
        """
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        self.mean_ = X.mean(axis=0)
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))

        for c in self.classes_:
            Xc = X[y == c]
            mu_c = Xc.mean(axis=0)
            Xc_centrada = Xc - mu_c
            Sw += Xc_centrada.T @ Xc_centrada
            d = (mu_c - self.mean_).reshape(-1, 1)
            Sb += len(Xc) * (d @ d.T)

        Sw += self.reg * np.trace(Sw) / n_features * np.eye(n_features)

        # Problema de autovalores generalizado simétrico: Sb v = λ Sw v
        valores, vectores = eigh(Sb, Sw)
        orden = np.argsort(valores)[::-1]
        valores, vectores = valores[orden], vectores[:, orden]

        max_comp = min(len(self.classes_) - 1, n_features)
        k = max_comp if self.n_components is None else min(self.n_components, max_comp)

        self.eigenvalues_ = valores[:k]
        self.explained_ratio_ = valores[:k] / valores[:max_comp].sum()
        self.components_ = vectores[:, :k]  # (n_features, k)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
