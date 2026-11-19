"""K-Nearest Neighbors con distancias totalmente vectorizadas en NumPy."""
import numpy as np

# Con métricas enteras hay muchísimos empates exactos de distancia, y el cálculo
# con BLAS rompe esos empates con ruido de redondeo que puede cambiar entre
# ejecuciones. El desempate determinista redondea las distancias y resuelve los
# empates por índice, de modo que el resultado no depende de ese ruido.
DESEMPATE_DETERMINISTA = True
_DECIMALES = 6


class KNNClassifier:
    def __init__(self, k=5, batch_size=1000, desempate=None):
        """
        k : int
            Número de vecinos.
        batch_size : int
            Cuántos puntos de consulta se procesan a la vez. Evita construir la
            matriz de distancias completa (n_consulta x n_entrenamiento) en memoria.
        desempate : bool | None
            Si es True, las distancias se redondean y los empates se resuelven por
            índice (reproducible). None usa DESEMPATE_DETERMINISTA.
        """
        self.k = k
        self.batch_size = batch_size
        self.desempate = DESEMPATE_DETERMINISTA if desempate is None else desempate

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.classes_, self.y_ = np.unique(np.asarray(y), return_inverse=True)
        self._sq_norms = (self.X_ ** 2).sum(axis=1)
        # Sesgo por índice, menor que la mitad de la rejilla de redondeo
        self._sesgo = np.arange(len(self.X_)) * (0.5 * 10.0 ** -_DECIMALES / max(len(self.X_), 1))
        return self

    def kneighbors(self, X, k=None):
        """Devuelve (distancias, índices) de los k vecinos más cercanos, ordenados
        de menor a mayor distancia. Calcular con un k grande y recortar columnas
        permite evaluar varios valores de k con un solo cálculo."""
        k = self.k if k is None else k
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        dist_out = np.empty((n, k))
        idx_out = np.empty((n, k), dtype=int)

        for ini in range(0, n, self.batch_size):
            B = X[ini:ini + self.batch_size]
            # ||a - b||² = ||a||² + ||b||² - 2 a·b  (sin bucles anidados)
            d2 = (B ** 2).sum(axis=1)[:, None] + self._sq_norms[None, :] - 2.0 * (B @ self.X_.T)
            np.maximum(d2, 0, out=d2)
            if self.desempate:
                np.round(d2, _DECIMALES, out=d2)
                d2 += self._sesgo[None, :]

            cand = np.argpartition(d2, k - 1, axis=1)[:, :k]
            d_cand = np.take_along_axis(d2, cand, axis=1)
            orden = np.argsort(d_cand, axis=1)

            idx_out[ini:ini + len(B)] = np.take_along_axis(cand, orden, axis=1)
            dist_out[ini:ini + len(B)] = np.sqrt(np.take_along_axis(d_cand, orden, axis=1))

        return dist_out, idx_out

    def predict_proba_from_neighbors(self, idx):
        """Proporción de votos por clase a partir de los índices de vecinos."""
        etiquetas = self.y_[idx]                                   # (n, k)
        votos = (etiquetas[:, :, None] == np.arange(len(self.classes_))).sum(axis=1)
        return votos / idx.shape[1]

    def predict_from_neighbors(self, idx):
        """Voto mayoritario. En caso de empate gana la clase de menor índice."""
        return self.classes_[self.predict_proba_from_neighbors(idx).argmax(axis=1)]

    def predict_proba(self, X):
        _, idx = self.kneighbors(X)
        return self.predict_proba_from_neighbors(idx)

    def predict(self, X):
        _, idx = self.kneighbors(X)
        return self.predict_from_neighbors(idx)
