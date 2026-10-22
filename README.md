# Clasificador de Posiciones de Fútbol

Proyecto de Machine Learning que, dadas las métricas de un jugador de fútbol, determina cuál es su posición más apta en el campo.

> **Estado:** Fase 1 completada (datos). Fase 2 en desarrollo: Etapa 0 y Nivel 1 listos; falta el Nivel 2.

## Arquitectura

El problema se aborda como una **cascada de tres etapas**:

0. **Etapa 0: portero vs. jugador de campo.** Un árbol de decisión de profundidad 1 sobre las métricas `goalkeeping_*`. Los porteros terminan aquí.
1. **Nivel 1: posición general de campo** (`Defensa`, `Medio`, `Delantero`). Proyección con **Análisis Discriminante de Fisher** (2 discriminantes) y clasificación con **KNN**.
2. **Nivel 2: posición específica**, con un submodelo por bloque (2A defensas, 2B medios, 2C delanteros). Cada uno usa **SFS** para seleccionar métricas y KNN para clasificar.

```
Jugador → Etapa 0: ¿portero o campo?
            ├─ Portero → fin
            └─ Campo → Nivel 1 (Fisher + KNN) → Nivel 2 (SFS + KNN) → posición específica
```

Fisher y KNN están implementados desde cero con NumPy/SciPy (`src/models/`) y validados contra scikit-learn.

## Datos

Para validar la arquitectura se usa el dataset de **EA Sports FC 24** (Kaggle):

- [`stefanoleone992/ea-sports-fc-24-complete-player-dataset`](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)

El archivo original contiene varias versiones del juego, por lo que un mismo jugador aparece repetido (unas 180,000 filas). El preprocesamiento conserva un único registro por jugador, lo que deja **18,350 jugadores** y **43 métricas** de entrada (atributos en escala de 0 a 100, más estatura y peso).

Una vez validado el pipeline, el dataset podrá sustituirse por estadísticas reales por 90 minutos (por ejemplo, de FBref) y reentrenar los modelos.

## Estructura del proyecto

```
clasificador-posiciones-futbol/
├── data/
│   ├── raw/          # Datos crudos descargados de Kaggle (no versionados)
│   └── processed/    # Matrices X e y listas para entrenar (no versionadas)
├── notebooks/
│   ├── 01_exploracion.ipynb      # Análisis exploratorio
│   └── 02_nivel1_fisher.ipynb    # Etapa 0 y Nivel 1
├── src/
│   ├── data/
│   │   └── make_dataset.py       # Descarga, limpieza y mapeo de posiciones
│   └── models/
│       ├── fisher.py             # Análisis Discriminante de Fisher
│       └── knn.py                # KNN vectorizado
├── environment.yml
└── README.md
```

## Requisitos

- [Anaconda](https://www.anaconda.com/) o Miniconda
- Cuenta de Kaggle con un token de API

Librerías principales: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyterlab` y `kaggle`.

## Configuración

### 1. Crear el entorno

Desde la raíz del repositorio:

```bash
conda env create -f environment.yml
conda activate futbol
```

### 2. Credenciales de Kaggle

Crea un token desde *Settings → API → Create New Token* en Kaggle. Según la versión de la plataforma, recibirás uno de estos dos formatos.

**Archivo `kaggle.json`** (formato clásico). Muévelo a la carpeta de tu usuario.

En Windows (PowerShell):

```powershell
mkdir "$env:USERPROFILE\.kaggle" -ErrorAction SilentlyContinue
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

En Linux/macOS:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Token de texto que empieza con `KGAT_`** (formato nuevo). Guárdalo en un archivo `access_token` en la misma carpeta, o en la variable de entorno `KAGGLE_API_TOKEN`.

En Windows (PowerShell):

```powershell
mkdir "$env:USERPROFILE\.kaggle" -ErrorAction SilentlyContinue
Set-Content -Path "$env:USERPROFILE\.kaggle\access_token" -Value "KGAT_pega_tu_token_aqui" -NoNewline
```

En Linux/macOS:

```bash
mkdir -p ~/.kaggle
echo -n "KGAT_pega_tu_token_aqui" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Para comprobar que la autenticación funciona:

```bash
kaggle datasets list -s "ea sports fc 24"
```

> Los tokens son credenciales privadas y nunca deben subirse al repositorio.

### 3. Obtener y procesar los datos

```bash
python src/data/make_dataset.py
```

Descarga el dataset a `data/raw/` (se omite si ya existe) y genera `data/processed/jugadores_procesados.csv`.

### 4. Cómo abrir los notebooks

```bash
conda activate futbol
jupyter lab
```

Ejecútalo desde la raíz del repositorio y abre los notebooks de la carpeta `notebooks/` en orden.

## Hoja de ruta

- [x] Definir la arquitectura del ensamble
- [x] Inicializar el repositorio y su estructura
- [x] **Fase 1:** ingesta y preprocesamiento de datos
  - [x] Descarga automatizada con la API de Kaggle
  - [x] Aislar la posición principal de cada jugador
  - [x] Mapear las posiciones específicas a las 4 categorías generales
  - [x] Exportar las matrices `X` e `y`
- [ ] **Fase 2:** modelado
  - [x] Etapa 0: árbol de decisión que separa porteros y jugadores de campo
  - [x] Nivel 1: Fisher + KNN para la posición general
  - [ ] Nivel 2: SFS + KNN para cada bloque
  - [ ] Evaluación del ensamble completo
- [ ] **Fase 3:** reemplazar los datos de EA FC por estadísticas reales (FBref)

## Licencia

Por definir.