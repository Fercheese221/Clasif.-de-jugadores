"""Fase 1: descarga y preprocesamiento del dataset de EA FC 24.

Uso (desde cualquier carpeta):
    python src/data/make_dataset.py
"""
from pathlib import Path

import pandas as pd

DATASET = "stefanoleone992/ea-sports-fc-24-complete-player-dataset"
RAW_FILE = "male_players.csv"

# Rutas ancladas a la raíz del repo, no al directorio desde el que se ejecute
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

MACRO = {
    "GK": "Portero",
    "CB": "Defensa", "RB": "Defensa", "LB": "Defensa", "RWB": "Defensa", "LWB": "Defensa",
    "CM": "Medio", "CDM": "Medio", "CAM": "Medio", "RM": "Medio", "LM": "Medio",
    "ST": "Delantero", "CF": "Delantero", "RW": "Delantero", "LW": "Delantero",
}

# Métricas de entrada. Se excluyen a propósito las columnas de calificación por
# posición (ls, st, rs, cb, gk...) porque filtrarían la respuesta al modelo.
COMPUESTAS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
PREFIJOS = ("attacking_", "skill_", "movement_", "power_", "mentality_",
            "defending_", "goalkeeping_")
FISICAS = ["height_cm", "weight_kg"]
EXCLUIR = {"skill_moves"}  # escala 1-5, no 0-100


def download_data(force=False):
    """Descarga y extrae el dataset desde Kaggle (omite si ya existe)."""
    destino = RAW_DIR / RAW_FILE
    if destino.exists() and not force:
        print(f"{RAW_FILE} ya existe, se omite la descarga.")
        return

    from kaggle.api.kaggle_api_extended import KaggleApi  # import tardío

    print("Iniciando descarga desde Kaggle...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=True)
    print("Dataset descargado y extraído.")


def process_data():
    """Limpia posiciones y genera la variable objetivo del Nivel 1."""
    print("Procesando la matriz de jugadores...")
    df = pd.read_csv(RAW_DIR / RAW_FILE, low_memory=False)
    print(f"Filas en el archivo original: {len(df):,}")

    # 1. Un solo registro por jugador: el archivo puede traer varias versiones
    #    del juego y varias actualizaciones, y duplicar jugadores contaminaría
    #    la evaluación (el mismo jugador en entrenamiento y en prueba).
    if "fifa_version" in df.columns:
        df = df[df["fifa_version"] == df["fifa_version"].max()]
    if "fifa_update" in df.columns:
        df = df.sort_values("fifa_update")
    df = df.drop_duplicates("player_id", keep="last").copy()
    print(f"Jugadores únicos tras filtrar: {len(df):,}")

    # 2. Posición principal (primera de la lista)
    df["posicion_principal"] = df["player_positions"].str.split(",").str[0].str.strip()

    # 3. Mapeo a las 4 clases macro; .map deja NaN en lo no mapeado
    df["y_gen"] = df["posicion_principal"].map(MACRO)
    sin_mapear = df["y_gen"].isna()
    if sin_mapear.any():
        raros = df.loc[sin_mapear, "posicion_principal"].unique()
        print(f"Aviso: se descartan {sin_mapear.sum()} filas con posiciones sin mapear: {raros}")
        df = df[~sin_mapear]

    # 4. Selección de columnas de entrada
    metricas = [c for c in COMPUESTAS if c in df.columns]
    metricas += [c for c in df.columns if c.startswith(PREFIJOS) and c not in EXCLUIR]
    metricas += [c for c in FISICAS if c in df.columns]

    df_out = df[["player_id", "short_name", "posicion_principal", "y_gen"] + metricas].copy()

    # 5. Nulos estructurales: los porteros no tienen métricas de campo (pace,
    #    shooting...) y los jugadores de campo no tienen las de portero.
    #    Se rellenan con 0 por ahora; en el Nivel 2 conviene descartar las
    #    columnas goalkeeping_* para los bloques que no sean porteros.
    nulos_restantes = df_out[metricas].isna().sum().sum()
    print(f"Celdas nulas en métricas (se rellenan con 0): {nulos_restantes:,}")
    df_out[metricas] = df_out[metricas].fillna(0)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    salida = PROCESSED_DIR / "jugadores_procesados.csv"
    df_out.to_csv(salida, index=False)

    print(f"Matriz guardada en: {salida}")
    print(f"Dimensiones: {df_out.shape} ({len(metricas)} métricas)")
    print("\nDistribución de y_gen:")
    print(df_out["y_gen"].value_counts())


if __name__ == "__main__":
    download_data()
    process_data()