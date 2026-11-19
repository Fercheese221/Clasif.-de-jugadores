"""Verifica si los resultados cambian entre ejecuciones y por qué.

Uso, desde la raíz del repositorio:

  python scripts/verificar_reproducibilidad.py sensibilidad
  python scripts/verificar_reproducibilidad.py correr resultados/run1
  python scripts/verificar_reproducibilidad.py correr resultados/run2
  python scripts/verificar_reproducibilidad.py comparar resultados/run1 resultados/run2

  Añade --sin-desempate a `correr` para reproducir el comportamiento anterior
  del KNN (sin el desempate determinista).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402

import src.models.knn as knn_mod                      # noqa: E402
from src.models.cascada import Cascada                # noqa: E402
from src.models.knn import KNNClassifier              # noqa: E402
from src.taxonomia import BLOQUES, MAPA_CLASE, ORDEN  # noqa: E402

SEED = 42
ID_COLS = ["player_id", "short_name", "posicion_principal", "y_gen", "player_positions"]


def cargar():
    df = pd.read_csv(ROOT / "data" / "processed" / "jugadores_procesados.csv")
    metricas = [c for c in df.columns if c not in ID_COLS]
    metricas_campo = [c for c in metricas if not c.startswith("goalkeeping_")]
    df["clase"] = df["posicion_principal"].map(MAPA_CLASE)
    train, test = train_test_split(df, test_size=0.2, stratify=df["posicion_principal"], random_state=SEED)
    tr = train[train["y_gen"] != "Portero"].reset_index(drop=True)
    te = test[test["y_gen"] != "Portero"].reset_index(drop=True)
    return (tr[metricas_campo].to_numpy(float), tr["clase"].to_numpy(),
            te[metricas_campo].to_numpy(float), te["clase"].to_numpy(), metricas_campo)


def info_entorno():
    info = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        from threadpoolctl import threadpool_info
        info["blas"] = [{"api": d.get("internal_api"), "version": d.get("version"),
                         "hilos": d.get("num_threads")} for d in threadpool_info()]
    except Exception:
        info["blas"] = "no disponible"
    return info


# ---------------------------------------------------------------- sensibilidad
def sensibilidad(_args):
    print("Entorno:", json.dumps(info_entorno(), ensure_ascii=False))
    X_tr, y_tr, X_te, _, _ = cargar()
    sc = StandardScaler().fit(X_tr)
    A, Q = sc.transform(X_tr), sc.transform(X_te)

    print("\n1) ¿La multiplicación de matrices (BLAS) da resultados idénticos con distinta alineación de memoria?")
    ref = Q[:1000] @ A.T
    distintos = 0
    for desplazamiento in range(8):
        buf = np.empty(A.size + 8)
        A2 = buf[desplazamiento:desplazamiento + A.size].reshape(A.shape)
        A2[:] = A
        distintos += int((Q[:1000] @ A2.T != ref).sum())
    print(f"   Celdas distintas en 8 pruebas: {distintos}  "
          f"({'idéntico' if distintos == 0 else 'NO es bit a bit reproducible'})")

    print("\n2) Sensibilidad del KNN a ruido numérico de 1e-13 (k = 15)")
    print("   Compara dos KNN con datos que difieren ~1e-13, con y sin desempate determinista.")
    rng = np.random.default_rng(0)
    orden_cols = rng.permutation(A.shape[1])
    print(f"   {'métricas':>9} | {'vecindarios distintos':>26} | {'predicciones distintas':>26}")
    print(f"   {'':>9} | {'sin → con desempate':>26} | {'sin → con desempate':>26}")
    for d in (1, 3, 10, A.shape[1]):
        a, q = A[:, orden_cols[:d]], Q[:, orden_cols[:d]]
        a2 = a * (1 + 1e-13 * rng.standard_normal(a.shape))
        q2 = q * (1 + 1e-13 * rng.standard_normal(q.shape))
        res = []
        for des in (False, True):
            m1 = KNNClassifier(k=15, desempate=des).fit(a, y_tr)
            m2 = KNNClassifier(k=15, desempate=des).fit(a2, y_tr)
            _, i1 = m1.kneighbors(q)
            _, i2 = m2.kneighbors(q2)
            res.append(((np.sort(i1, 1) != np.sort(i2, 1)).any(axis=1).mean(),
                        (m1.predict_from_neighbors(i1) != m2.predict_from_neighbors(i2)).mean()))
        print(f"   {d:>9} | {res[0][0]:>12.1%} → {res[1][0]:>9.1%} | {res[0][1]:>12.1%} → {res[1][1]:>9.1%}")

    print("\nCómo leerlo: si sin desempate hay vecindarios/predicciones distintos con pocas métricas, "
          "SFS (que empieza con 1 a 3 métricas) es sensible al ruido numérico. "
          "Con desempate, ambos deben ser 0%.")


# ---------------------------------------------------------------------- correr
def evaluar(P, y_idx):
    n = len(y_idx)
    orden = np.argsort(-P, axis=1, kind="stable")   # desempate reproducible
    Pc = np.clip(P, 1e-3, 1.0)
    Pc = Pc / Pc.sum(axis=1, keepdims=True)
    return {
        "top-1": float((orden[:, 0] == y_idx).mean()),
        "top-2": float((orden[:, :2] == y_idx[:, None]).any(axis=1).mean()),
        "top-3": float((orden[:, :3] == y_idx[:, None]).any(axis=1).mean()),
        "log-loss": float(-np.log(Pc[np.arange(n), y_idx]).mean()),
    }


def correr(args):
    knn_mod.DESEMPATE_DETERMINISTA = not args.sin_desempate
    X_tr, y_tr, X_te, y_te, metricas_campo = cargar()
    print(f"Ajustando la cascada A (desempate {'desactivado' if args.sin_desempate else 'activado'})...")

    casc = Cascada(BLOQUES["A"], orden_clases=ORDEN,
                   sfs_kwargs=dict(n_splits=3, max_features=args.max_features))
    casc.fit(X_tr, y_tr, nombres=metricas_campo, verbose=False)
    P = casc.predict_proba(X_te, ruta="suave")
    y_idx = np.array([ORDEN.index(c) for c in y_te])

    resumen = {
        "entorno": info_entorno(),
        "desempate": not args.sin_desempate,
        "k1": int(casc.k1_),
        "bloques": {
            str(b): {"k": (None if m.k_ is None else int(m.k_)),
                     "metricas": list(getattr(m, "metricas_seleccionadas_", []))}
            for b, m in casc.nivel2_.items()
        },
        "metricas": evaluar(P, y_idx),
    }
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{salida}_probas.npy", P)
    Path(f"{salida}.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"k1": resumen["k1"], "metricas": resumen["metricas"]}, ensure_ascii=False))
    print(f"Guardado en {salida}.json y {salida}_probas.npy")


# -------------------------------------------------------------------- comparar
def comparar(args):
    runs = []
    for prefijo in args.archivos:
        datos = json.loads(Path(f"{prefijo}.json").read_text(encoding="utf-8"))
        datos["P"] = np.load(f"{prefijo}_probas.npy")
        datos["nombre"] = Path(prefijo).name
        runs.append(datos)

    base = runs[0]
    print(f"Referencia: {base['nombre']} (desempate {'activado' if base['desempate'] else 'desactivado'})")
    todo_igual = True
    for r in runs[1:]:
        print(f"\n--- {r['nombre']} vs {base['nombre']} ---")
        print(f"k del Nivel 1: {base['k1']} → {r['k1']}" + ("" if base["k1"] == r["k1"] else "   ← DISTINTO"))
        for b in base["bloques"]:
            mb, mr = base["bloques"][b], r["bloques"][b]
            igual = mb["metricas"] == mr["metricas"] and mb["k"] == mr["k"]
            print(f"  {b}: k {mb['k']} → {mr['k']}, métricas {'iguales' if igual else 'DISTINTAS'}")
            if mb["metricas"] != mr["metricas"]:
                print(f"     antes:  {mb['metricas']}")
                print(f"     ahora:  {mr['metricas']}")
        dif = np.abs(base["P"] - r["P"])
        top1_dif = int((base["P"].argmax(1) != r["P"].argmax(1)).sum())
        print(f"Diferencia máxima en probabilidades: {dif.max():.2e}")
        print(f"Jugadores con top-1 distinto: {top1_dif} de {len(base['P'])}")
        for m in base["metricas"]:
            print(f"  {m:>9}: {base['metricas'][m]:.4f} → {r['metricas'][m]:.4f}")
        todo_igual &= bool(dif.max() == 0)

    print("\nVEREDICTO:", "las ejecuciones son IDÉNTICAS" if todo_igual else "las ejecuciones DIFIEREN")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sensibilidad").set_defaults(func=sensibilidad)

    p = sub.add_parser("correr")
    p.add_argument("salida", help="prefijo de los archivos de salida, p. ej. resultados/run1")
    p.add_argument("--sin-desempate", action="store_true")
    p.add_argument("--max-features", type=int, default=15)
    p.set_defaults(func=correr)

    p = sub.add_parser("comparar")
    p.add_argument("archivos", nargs="+", help="prefijos de las ejecuciones a comparar (la primera es la referencia)")
    p.set_defaults(func=comparar)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()