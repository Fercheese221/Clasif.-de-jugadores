# Clasificador de Posiciones de Fútbol

Proyecto de Machine Learning que, dadas las métricas de un jugador de fútbol, determina cuál es su posición más apta en el campo.

> **Estado:** Fase 1 en desarrollo (ingesta y preprocesamiento de datos).

## Arquitectura

El problema se aborda como una clasificación multiclase mediante un **ensamble jerárquico en cascada**:

1. **Nivel 1 – Posición general.** Clasifica al jugador en `Portero`, `Defensa`, `Medio` o `Delantero`. Antes de clasificar, los datos se proyectan con **Análisis Discriminante de Fisher**, que maximiza la separación entre clases.
2. **Nivel 2 – Posición específica.** Un submodelo especialista por cada bloque:
   - **2A (Defensas):** Central, Lateral.
   - **2B (Medios):** Pivote defensivo, Interior, Mediapunta.
   - **2C (Delanteros):** Extremo, Falso 9, Nueve de área.

   Cada submodelo usa **Sequential Feature Selection (SFS)** para quedarse solo con las métricas relevantes de su bloque.
3. **Clasificador final.** **K-Nearest Neighbors (KNN)** con el cálculo de distancias vectorizado en NumPy, sin bucles anidados.

```
Vector del jugador → Fisher + Modelo 1 → Posición general
                   → Filtro SFS del bloque → Modelo 2 (KNN) → Posición específica
```

## Datos

Para validar la arquitectura se usa el dataset de **EA Sports FC 24** (Kaggle), con más de 18,000 jugadores y unas 40 métricas en escala de 0 a 100:

- [`stefanoleone992/ea-sports-fc-24-complete-player-dataset`](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)

Una vez validado el pipeline, el dataset podrá sustituirse por estadísticas reales por 90 minutos (por ejemplo, de FBref) y reentrenar los modelos.

## Estructura del proyecto

```
clasificador-posiciones-futbol/
├── data/
│   ├── raw/          # Datos crudos descargados de Kaggle (no versionados)
│   └── processed/    # Matrices X e y listas para entrenar (no versionadas)
├── notebooks/        # Exploración y experimentos
├── src/
│   └── data/
│       └── make_dataset.py   # Descarga, limpieza y mapeo de posiciones
└── README.md
```

## Requisitos

- Python 3.10 o superior
- Cuenta de Kaggle con un token de API (`kaggle.json`)

Librerías previstas: `pandas`, `numpy`, `scikit-learn`, `kaggle`.

## Configuración

### 1. Credenciales de Kaggle

Descarga tu `kaggle.json` desde *Settings → API → Create New Token* en Kaggle y colócalo en la carpeta de tu usuario.

En Windows (PowerShell):

```powershell
pip install kaggle
mkdir "$env:USERPROFILE\.kaggle" -ErrorAction SilentlyContinue
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

En Linux/macOS:

```bash
pip install kaggle
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

> El archivo `kaggle.json` contiene llaves privadas y nunca debe subirse al repositorio.

### 2. Obtener los datos

```bash
kaggle datasets download -d stefanoleone992/ea-sports-fc-24-complete-player-dataset --unzip -p data/raw
```

## Hoja de ruta

- [x] Definir la arquitectura del ensamble
- [x] Inicializar el repositorio y su estructura
- [ ] **Fase 1:** ingesta y preprocesamiento de datos
  - [ ] Descarga automatizada con la API de Kaggle
  - [ ] Aislar la posición principal de cada jugador
  - [ ] Mapear las posiciones específicas a las 4 categorías generales
  - [ ] Exportar las matrices `X` e `y`
- [ ] **Fase 2:** modelado
  - [ ] Nivel 1: Fisher + KNN para la posición general
  - [ ] Nivel 2: SFS + KNN para cada bloque
  - [ ] Evaluación del ensamble completo
- [ ] **Fase 3:** reemplazar los datos de EA FC por estadísticas reales (FBref)

## Licencia

Por definir.