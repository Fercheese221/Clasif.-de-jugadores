"""Cascada de dos niveles con enrutamiento suave.

Nivel 1 clasifica al jugador en un bloque (p. ej. Defensa / Medio / Delantero).
Nivel 2 tiene un submodelo por bloque que clasifica la posición específica.

  P(clase) = sum_b  P(bloque = b | x) * P(clase | x, bloque = b)

Con ruta="dura" el jugador solo pasa por el bloque más probable.
La taxonomía es un parámetro: `clase_a_bloque` indica a qué bloque pertenece
cada clase específica, así que la misma cascada sirve para taxonomías distintas.
"""
import numpy as np

from .metricas import elegir_k
from .nivel1 import FisherKNN, curva_k_fisher
from .sfs import SFSKNN, ClaseUnica


class Cascada:
    def __init__(self, clase_a_bloque, orden_clases=None, k1=None, ks=None, sfs_kwargs=None, seed=42):
        self.clase_a_bloque = dict(clase_a_bloque)
        self.orden_clases = orden_clases
        self.k1 = k1
        self.ks = list(range(1, 52, 2)) if ks is None else list(ks)
        self.sfs_kwargs = dict(sfs_kwargs or {})
        self.seed = seed

    def fit(self, X, y_clase, nombres=None, verbose=True):
        X = np.asarray(X, dtype=float)
        y_clase = np.asarray(y_clase)

        presentes = set(y_clase)
        desconocidas = presentes - set(self.clase_a_bloque)
        if desconocidas:
            raise ValueError(f"Clases sin bloque asignado: {sorted(desconocidas)}")

        if self.orden_clases is None:
            self.classes_ = sorted(presentes)
        else:
            self.classes_ = [c for c in self.orden_clases if c in presentes]
        self.col_ = {c: i for i, c in enumerate(self.classes_)}

        y_bloque = np.array([self.clase_a_bloque[c] for c in y_clase])

        # Nivel 1
        if self.k1 is None:
            curva = curva_k_fisher(X, y_bloque, self.ks, seed=self.seed)
            self.k1_ = elegir_k(curva)
            self.curva_k1_ = curva
        else:
            self.k1_ = self.k1
        self.nivel1_ = FisherKNN(k=self.k1_).fit(X, y_bloque)
        if verbose:
            print(f"  Nivel 1 · bloques {[str(b) for b in self.nivel1_.classes_]} · k = {self.k1_}")

        # Nivel 2
        self.nivel2_ = {}
        for b in self.nivel1_.classes_:
            m = y_bloque == b
            clases_b = np.unique(y_clase[m])
            if len(clases_b) == 1:
                self.nivel2_[b] = ClaseUnica(clases_b[0])
                if verbose:
                    print(f"  Nivel 2 · {b}: una sola clase ({clases_b[0]}), sin submodelo")
            else:
                if verbose:
                    print(f"  Nivel 2 · {b}: {len(clases_b)} clases, {m.sum():,} jugadores → SFS")
                kw = dict(self.sfs_kwargs, verbose=verbose, seed=self.seed)
                self.nivel2_[b] = SFSKNN(**kw).fit(X[m], y_clase[m], nombres=nombres)
                if verbose:
                    print(f"    k = {self.nivel2_[b].k_} · {len(self.nivel2_[b].selected_)} métricas")
        return self

    def predict_proba(self, X, ruta="suave"):
        X = np.asarray(X, dtype=float)
        P1 = self.nivel1_.predict_proba(X)
        if ruta == "dura":
            duro = np.zeros_like(P1)
            duro[np.arange(len(P1)), P1.argmax(axis=1)] = 1.0
            P1 = duro
        elif ruta != "suave":
            raise ValueError("ruta debe ser 'suave' o 'dura'")

        P = np.zeros((len(X), len(self.classes_)))
        for j, b in enumerate(self.nivel1_.classes_):
            modelo = self.nivel2_[b]
            P2 = modelo.predict_proba(X)
            for i, c in enumerate(modelo.classes_):
                P[:, self.col_[c]] += P1[:, j] * P2[:, i]
        return P

    def predict(self, X, ruta="suave"):
        return np.array(self.classes_)[self.predict_proba(X, ruta).argmax(axis=1)]
