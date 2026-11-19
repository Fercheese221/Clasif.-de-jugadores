# Clasificador de Posiciones de Fútbol

Proyecto de Machine Learning que, dadas las métricas de un jugador de fútbol, determina cuál es su posición más apta en el campo. La salida es una **distribución de probabilidad sobre posiciones**, no una única etiqueta.

> **Estado:** Fase 1 completada (datos). Fase 2: Etapa 0, Nivel 1 y Nivel 2 implementados, evaluados y verificados como reproducibles; falta empaquetar la función de predicción. La Fase 3 (datos reales) debe replantearse.

## Arquitectura

El problema se aborda como una **cascada** con enrutamiento suave:

0. **Etapa 0: portero vs. jugador de campo.** Un árbol de decisión de profundidad 1 sobre las métricas `goalkeeping_*`. Los porteros terminan aquí.
1. **Nivel 1: bloque de campo** (`Defensa`, `Medio`, `Delantero`). Proyección con **Análisis Discriminante de Fisher** y clasificación con **KNN**.
2. **Nivel 2: posición específica**, con un submodelo por bloque. Cada uno usa **SFS** (selección secuencial de métricas) y **KNN**.

Las **9 clases finales** son: Central, Lateral, Carrilero, Pivote, Interior, Mediapunta, Mediocampista de banda, Extremo y Delantero centro. Los pares izquierda/derecha están fusionados (las métricas no contienen el lado del campo) y CF se funde con ST.

```
Jugador → Etapa 0: ¿portero o campo?
            ├─ Portero → fin
            └─ Campo → Nivel 1 → P(bloque)
                       Nivel 2 → P(clase | bloque)   (un submodelo por bloque)

P(clase) = Σ_bloque  P(bloque) · P(clase | bloque)
```

Con el **enrutamiento suave**, el jugador pasa por todos los bloques ponderado por su probabilidad, de modo que un error del Nivel 1 deja de ser definitivo. El enrutamiento duro (solo el bloque más probable) se conserva para comparar.

La taxonomía es un parámetro: `Cascada` recibe qué clase pertenece a qué bloque (`src/taxonomia.py`). Se evaluaron dos agrupaciones: **A** (Defensa / Medio / Delantero), que es la configuración por defecto, y **B** (con un bloque adicional `Banda`), que se descartó por ser más frágil (ver Resultados).

Fisher, KNN, SFS y la cascada están implementados desde cero con NumPy/SciPy (`src/models/`) y validados contra scikit-learn.

## Datos

Para validar la arquitectura se usa el dataset de **EA Sports FC 24** (Kaggle):

- [`stefanoleone992/ea-sports-fc-24-complete-player-dataset`](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset)

El archivo original contiene varias versiones del juego, por lo que un mismo jugador aparece repetido (unas 180,000 filas). El preprocesamiento conserva un único registro por jugador, lo que deja **18,350 jugadores** y **43 métricas** de entrada (atributos en escala de 0 a 100, más estatura y peso).

Las etiquetas de EA FC son calificaciones de un videojuego, no comportamiento real en el campo. Esto limita lo que se puede distinguir (ver Resultados).

## Resultados hasta ahora

Evaluados sobre un conjunto de prueba de 3,670 jugadores (3,261 de campo), con la misma partición estratificada en todos los notebooks. Los resultados son reproducibles entre ejecuciones (ver Reproducibilidad); las diferencias de 1 a 2 puntos entre modelos están en el orden del error de muestreo (error estándar del top-1 ≈ 0.8 puntos).

| Componente | Resultado |
|---|---|
| Etapa 0 (portero vs. campo) | 100% de exactitud (409 porteros, sin errores) |
| Nivel 1 aislado (3 bloques de campo) | Exactitud ≈ 0.86 (clase mayoritaria: 0.41). Fisher comprime 37 métricas en 2 dimensiones y empata con KNN sin proyección |
| Cascada completa, 9 clases (enrutamiento suave, taxonomía A) | top-1 ≈ 0.68, top-2 ≈ 0.86, top-3 ≈ 0.94; acierto ≈ 0.82 si se aceptan las posiciones secundarias del jugador |
| Enrutamiento suave vs. duro | El log-loss baja ≈ 39% (1.46 → 0.89) y el top-3 sube de 0.86 a 0.94; el top-1 no cambia |
| Cascada vs. modelos planos | No supera a los planos en top-1 (Fisher + KNN: 0.70; Random Forest: 0.73), aunque sí al KNN plano en log-loss (0.89 vs 0.98). Su valor es la interpretabilidad: cada bloque usa sus propias métricas |
| Taxonomía A vs. B (con bloque `Banda`) | A es igual o mejor (con 9 clases: mejor top-2, top-3 y log-loss; con 7 clases empatan). La selección de métricas de B cambia entre ejecuciones con desempeño equivalente y su calibración es peor. Se usa A |
| Clases fusionadas (7) | top-1 ≈ 0.76 (Random Forest: 0.79). Los jugadores de banda y los carrileros se ubican bien en su familia de rol |
| Calibración | La cascada A queda casi sobre la diagonal, con una ligera sobreconfianza en probabilidades altas. El Random Forest está subconfiado |
| AUC uno-contra-resto | Entre 0.89 y 0.99 por clase (el más bajo, Carrilero: 0.81) |
| Posiciones espejo (LB/RB, LW/RW...) | No se distinguen con estas métricas; se fusionan |

**Hallazgos principales**

- Hay dos tipos de error. Los **errores de prior**, como Mediapunta o Pivote frente a Interior: el modelo ordena bien (AUC ≈ 0.90) pero pierde el desempate por tamaño de clase, y el top-2 lo rescata. Y los **errores de información**, como Extremo vs. Mediocampista de banda (AUC 0.65) y Carrilero vs. Lateral (AUC 0.59): los atributos del videojuego casi no los distinguen.
- Por eso el top-1 subestima al modelo en las clases pequeñas, y conviene evaluar con top-2, top-3, AUC y log-loss.
- Combinar probabilidades entre niveles (enrutamiento suave) es lo que más mejora la calidad de las probabilidades.

## Reproducibilidad

Los resultados son deterministas: ejecuciones separadas dan probabilidades idénticas, también con distinto número de hilos de MKL. Dos decisiones lo garantizan:

- El KNN redondea las distancias y resuelve los empates por índice (`desempate`, activado por defecto). Con métricas enteras hay muchos empates de distancia, y sin este desempate el resultado sería frágil ante ruido numérico.
- La evaluación usa un orden estable (`np.argsort(..., kind="stable")`) para elegir la clase más probable. Con el orden por defecto de NumPy, un empate exacto entre dos clases se resolvía distinto según la función usada y producía discrepancias de hasta 1.4 puntos en el acierto de una posición.

Para verificarlo en otra máquina:

```bash
python scripts/verificar_reproducibilidad.py sensibilidad
python scripts/verificar_reproducibilidad.py correr resultados/run1
python scripts/verificar_reproducibilidad.py correr resultados/run2
python scripts/verificar_reproducibilidad.py comparar resultados/run1 resultados/run2
```

## Estructura del proyecto

```
clasificador-posiciones-futbol/
├── data/
│   ├── raw/          # Datos crudos descargados de Kaggle (no versionados)
│   └── processed/    # jugadores_procesados.csv y posiciones_jugadores.csv (no versionados)
├── notebooks/
│   ├── 01_exploracion.ipynb      # Análisis exploratorio
│   ├── 02_nivel1_fisher.ipynb    # Etapa 0 y Nivel 1
│   ├── 03_taxonomia.ipynb        # Taxonomías de clases, probabilidades y posiciones secundarias
│   └── 04_nivel2_cascada.ipynb   # Nivel 2, cascada con enrutamiento suave y evaluación completa
├── scripts/
│   └── verificar_reproducibilidad.py   # Comparación de resultados entre ejecuciones
├── src/
│   ├── taxonomia.py              # Posiciones -> clases finales -> bloques
│   ├── data/
│   │   └── make_dataset.py       # Descarga, limpieza y mapeo de posiciones
│   └── models/
│       ├── fisher.py             # Análisis Discriminante de Fisher
│       ├── knn.py                # KNN vectorizado con desempate determinista
│       ├── nivel1.py             # Escalado + Fisher + KNN (Nivel 1)
│       ├── sfs.py                # Selección secuencial de métricas + KNN (Nivel 2)
│       ├── cascada.py            # Cascada con enrutamiento suave
│       └── metricas.py           # Utilidades de evaluación
├── resultados/                   # Salidas de la verificación (no versionadas)
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

Descarga el dataset a `data/raw/` (se omite si ya existe) y genera en `data/processed/` los archivos `jugadores_procesados.csv` (métricas y etiquetas) y `posiciones_jugadores.csv` (lista completa de posiciones de cada jugador).

### 4. Cómo abrir los notebooks

```bash
conda activate futbol
jupyter lab
```

Ejecútalo desde la raíz del repositorio y abre los notebooks de la carpeta `notebooks/` en orden. El notebook 04 tarda varios minutos por la selección de métricas con validación cruzada.

## Hoja de ruta

- [x] Definir la arquitectura del ensamble
- [x] Inicializar el repositorio y su estructura
- [x] **Fase 1:** ingesta y preprocesamiento de datos
  - [x] Descarga automatizada con la API de Kaggle
  - [x] Aislar la posición principal de cada jugador
  - [x] Mapear las posiciones específicas a las categorías generales
  - [x] Exportar las matrices `X` e `y`
- [ ] **Fase 2:** modelado
  - [x] Etapa 0: árbol de decisión que separa porteros y jugadores de campo
  - [x] Nivel 1: Fisher + KNN para el bloque de campo
  - [x] Comparación de taxonomías de clases (se adopta la taxonomía A)
  - [x] Evaluación de las probabilidades de KNN (calibración, log-loss, top-k, AUC)
  - [x] Nivel 2: SFS + KNN por bloque, con enrutamiento suave
  - [x] Evaluación del ensamble completo frente a modelos planos
  - [x] Verificación de reproducibilidad entre ejecuciones
  - [ ] Función `predecir` que devuelva las posiciones más probables de un jugador, con criterio de desempate explícito, y guardado del modelo
- [ ] **Fase 3:** reemplazar los datos de EA FC por estadísticas reales
  - FBref retiró sus estadísticas avanzadas en enero de 2026, así que esta fase debe replantearse: hay que elegir una fuente que ofrezca métricas de comportamiento y posiciones detalladas

## Licencia

Por definir.