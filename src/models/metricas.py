"""Utilidades de evaluación compartidas por los modelos."""
import numpy as np


def f1_macro(y_true, y_pred, n_clases):
    """Macro-F1 para etiquetas enteras 0..n_clases-1 (más rápido que sklearn,
    útil dentro de bucles de selección de características)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.bincount(y_true * n_clases + y_pred, minlength=n_clases ** 2).reshape(n_clases, n_clases)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return f1.mean()


def elegir_k(curva, tolerancia=0.002):
    """Menor k cuyo puntaje esté a menos de `tolerancia` del máximo.
    `curva` es una Series con índice k y valor = puntaje."""
    return int(curva[curva >= curva.max() - tolerancia].index.min())
