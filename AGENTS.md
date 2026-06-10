# AGENTS.md

## Proposito de este archivo

Este archivo esta pensado para agentes, asistentes y contribuidores que necesiten trabajar rapido sobre este repositorio sin asumir una arquitectura que todavia no existe.

La idea principal es simple: hoy la logica real del proyecto vive en un notebook, no en una app Python modularizada.

## Mision del repositorio

Construir un pipeline reproducible para deteccion de danos en autos sobre CarDD usando PyTorch, empezando por la preparacion del dataset y dejando el camino listo para entrenamiento, evaluacion e inferencia futuras.

## Verdad operativa actual

- La fuente principal de verdad es `dev/01_dataset_preparation.ipynb`.
- `README.md` y `data/README.md` describen bien el objetivo, pero no reemplazan al notebook.
- `app.py` y `utils.py` no contienen logica util hoy.
- `prod/detection_dataset.py` ya contiene una implementacion activa y reutilizable del dataset de deteccion.
- El repo esta en etapa de preparacion de datos, no de producto final.

## Orden de lectura recomendado

1. `README.md`
2. `data/README.md`
3. `dev/01_dataset_preparation.ipynb`
4. `requirements.txt`
5. `.gitignore`

Si una duda no queda resuelta en los README, casi seguro la respuesta esta en el notebook.

## Mapa rapido: donde buscar cada cosa

### Proyecto y objetivo general

- `README.md`
- `CLAUDE.md`

### Dataset, estructura de carpetas y origen de los datos

- `data/README.md`
- `data/CarDD_release/CarDD_COCO/annotations/`
- `dev/01_dataset_preparation.ipynb`, seccion de descarga y deteccion del dataset

### Logica de descarga automatica

- `ensure_gdown`
- `download_cardd_zip`
- `extract_cardd_zip`
- `find_dataset_root`

### Logica de parsing de anotaciones COCO

- Seccion `## 4. Preparacion de anotaciones COCO para deteccion`

### Split train / val / test

- `split_records`
- `train_records`
- `val_records`
- `test_records`
- Seccion `## 5. Uso de splits oficiales train / val / test`

### Dataset y transforms para PyTorch

- `prod/detection_dataset.py`
- `ComposeDetection`
- `ToTensorDetection`
- `RandomHorizontalFlipDetection`
- `CarDamageDetectionDataset`

### DataLoaders

- `prod/detection_dataset.py`
- `collate_fn`
- `train_loader`
- `val_loader`
- `test_loader`

### Visualizacion y debugging visual

- `draw_boxes`
- `show_augmentations`
- `show_batch`

## Estructura real del repo

### Archivos raiz

- `README.md`: explica de que va el proyecto.
- `CLAUDE.md`: contexto tecnico amplio del repo.
- `AGENTS.md`: esta guia operativa.
- `requirements.txt`: dependencias.
- `.gitignore`: reglas para no versionar dataset pesado ni imagenes.
- `app.py`: placeholder vacio.
- `utils.py`: placeholder vacio.
- `prod/detection_dataset.py`: implementacion reutilizable del pipeline de datos para deteccion.

### Carpetas

- `dev/`: notebooks y trabajo exploratorio.
- `data/`: dataset local y documentacion de datos.
- `prod/`: codigo reusable fuera del notebook.

## Estado del dataset en este workspace

- Existe `data/CarDD_release/`.
- Dentro existe `CarDD_COCO/` y `CarDD_SOD/`.
- El pipeline actual solo usa `CarDD_COCO`.
- En Git no deben versionarse imagenes pesadas.
- En este workspace las anotaciones COCO si estan presentes.

## Punto importante sobre las carpetas de datos

No asumas que la estructura local siempre es identica. El notebook ya contempla varias rutas candidatas para encontrar el dataset.

Rutas que el notebook sabe detectar:

- `data/raw/CarDD/`
- `data/CarDD/`
- `data/CarDD_release/`
- `data/CarDD_release/CarDD_COCO/`
- `data/CarDD/CarDD_COCO/`

Si una tarea falla porque no encuentra datos, primero revisar `find_dataset_root()` antes de cambiar rutas a mano.

## Contrato conceptual del pipeline

### Entrada

- JSON COCO de train, val y test
- imagenes correspondientes en `train2017`, `val2017` y `test2017`

### Transformacion

- lectura de categorias, imagenes y annotations
- agrupacion por `image_id`
- conversion de boxes de `XYWH` a `XYXY`
- armado de `records` por imagen
- separacion por split oficial

### Salida actual

- listas `train_records`, `val_records`, `test_records`
- `Dataset` de PyTorch para deteccion reusable desde notebook o scripts
- `DataLoader`s listos para entrenamiento/evaluacion
- graficos y verificaciones visuales dentro del notebook

### Salida no implementada todavia

- CSVs persistidos a disco
- pipeline de entrenamiento versionado
- pipeline de evaluacion formal
- interfaz de inferencia

## Cuando te pidan algo, donde tocar

### Si te piden explicar el proyecto

- Basate primero en `README.md` y en `CLAUDE.md`.

### Si te piden arreglar o extender la preparacion del dataset

- Toca primero `dev/01_dataset_preparation.ipynb`.
- Si la logica ya esta madura, toca o extiende `prod/detection_dataset.py` y deja el notebook como consumidor de esa implementacion.

### Si te piden una funcion reutilizable

- No la dejes enterrada en el notebook si ya esta estabilizada.
- Por defecto, sumala a `prod/detection_dataset.py` o a un modulo nuevo dentro de `prod/`.

### Si te piden entrenamiento de modelo

- Hoy no existe implementacion base en archivos Python.
- Lo correcto es crearla sin romper el notebook actual, usando el notebook como referencia para datos y formato de targets.

### Si te piden una app o script de inferencia

- No asumas que `app.py` ya tiene estructura base. Hay que diseñarla desde cero o a partir de nuevos modulos.

## Hechos importantes que un agente no debe asumir mal

- Este repo no esta centrado en clasificacion simple; esta orientado a deteccion con bounding boxes.
- El dataset de verdad usado hoy es `CarDD_COCO`, no `CarDD_SOD`.
- El notebook crea `csv_manifest_df`, pero no guarda CSVs a disco en el estado actual.
- `README.md` menciona archivos que hoy no estan presentes; documenta eso en vez de asumir que faltan por error.
- `app.py` y `utils.py` no son la fuente de verdad actual.
- Para datos de deteccion, la implementacion reusable ahora esta en `prod/detection_dataset.py`.

## Checklist de trabajo seguro

1. Confirmar si la tarea impacta dataset, notebook, documentacion o futura modularizacion.
2. Leer la seccion relevante del notebook antes de editar nada.
3. No romper el contrato de `record` ni el formato de `target` si el cambio toca deteccion.
4. Mantener consistencia entre `README.md`, `data/README.md`, `CLAUDE.md` y `AGENTS.md`.
5. Si agregas persistencia de CSV o scripts de entrenamiento, dejar claramente documentado donde quedan y cual es la nueva fuente de verdad.

## Formato interno que conviene preservar

### `record`

Cada muestra del dataset se representa como un diccionario con:

- `image_path`
- `image_id`
- `boxes`
- `labels`
- `label_names`
- `area`
- `iscrowd`
- `split`
- `width`
- `height`

### `target`

El dataset devuelve un target compatible con modelos de deteccion de `torchvision`:

- `boxes`
- `labels`
- `image_id`
- `area`
- `iscrowd`

## Comandos base

Instalacion:

```bash
pip install -r requirements.txt
```

Trabajo interactivo:

```bash
jupyter notebook dev/01_dataset_preparation.ipynb
```

## Si tenes que responder rapido a "de que va este repo"

Respuesta corta correcta:

"Es un proyecto base para deteccion de danos en autos con CarDD y PyTorch. Hoy la parte implementada es la preparacion del dataset en formato COCO, con notebook, Dataset custom, transforms y DataLoaders; todavia no hay entrenamiento ni app final modularizados." 

## Regla final

Si algo parece faltar en el codigo Python plano, no concluyas enseguida que el repo esta incompleto por error. Primero revisa si esa logica ya vive en el notebook. En este proyecto, esa es la situacion normal.
