# CLAUDE.md

## Resumen del proyecto

Este repositorio es la base de un proyecto final de redes neuronales para deteccion de danos en autos usando el dataset CarDD.

El foco actual del repo no es una app final ni un pipeline de entrenamiento completo. El foco actual es dejar lista la capa de datos para una tarea de deteccion de objetos en PyTorch:

- descarga y deteccion automatica del dataset
- lectura de anotaciones COCO
- conversion de anotaciones a registros por imagen
- definicion de un `Dataset` custom para deteccion
- definicion de transforms compatibles con bounding boxes
- construccion de `DataLoader`s para train, val y test
- base de entrenamiento para tres variantes de Faster R-CNN
- evaluacion con `mAP@50:95` y `mAP@50`
- visualizacion y verificacion manual de batches y augmentations

Hoy, la fuente principal de verdad sigue siendo el notebook `dev/01_dataset_preparation.ipynb`, pero la implementacion reutilizable de datos ya se extrajo a `prod/detection_dataset.py`.

## Estado actual

- El notebook principal esta implementado y documenta el flujo de preparacion del dataset.
- `README.md` explica el objetivo general y como reproducir el entorno.
- `data/README.md` explica el dataset y la estructura esperada.
- `app.py` esta vacio.
- `utils.py` esta vacio.
- `prod/detection_dataset.py` contiene ahora la implementacion reutilizable del dataset de deteccion, transforms y `collate_fn`.
- `prod/detection_models.py`, `prod/detection_training.py` y `prod/detection_metrics.py` contienen la base reutilizable para la semana 3.
- No hay tests automaticos versionados.
- El repo menciona `data/train.csv`, `data/val.csv` y `data/test.csv`, pero en el estado actual esos archivos no estan presentes y el notebook tampoco los persiste a disco todavia.

## Objetivo tecnico actual

Preparar un pipeline reproducible para deteccion de danos en vehiculos usando CarDD en formato COCO, de forma que luego sea sencillo conectar un modelo de deteccion de PyTorch como Faster R-CNN, RetinaNet o similar.

## Dataset usado

- Dataset: CarDD, Car Damage Detection Dataset
- Fuente oficial: `https://cardd-ustc.github.io/`
- Variante usada en este repo: `CarDD_COCO`
- Descarga directa usada por el notebook: Google Drive via `gdown`
- Clases de dano detectadas en las anotaciones actuales: `dent`, `scratch`, `crack`, `glass shatter`, `lamp broken`, `tire flat`

El notebook arma tambien la clase `background` con indice `0` para compatibilidad con pipelines de deteccion.

## Estructura del repositorio

### Raiz

- `README.md`: overview del proyecto, stack, reproducibilidad local y Colab.
- `CLAUDE.md`: contexto general del repo, arquitectura actual y mapa funcional.
- `AGENTS.md`: guia operativa para agentes y contribuidores automatizados.
- `requirements.txt`: dependencias Python minimas del proyecto.
- `.gitignore`: excluye dataset pesado, imagenes, entornos y archivos temporales.
- `app.py`: reservado para una futura aplicacion o entrypoint; hoy esta vacio.
- `utils.py`: reservado para helpers reutilizables; hoy esta vacio.

### `dev/`

- `01_dataset_preparation.ipynb`: activo principal del repo.
- `02_model_training.ipynb`: notebook base para semana 3 con tres experimentos de Faster R-CNN.
- Si queres entender el pipeline de datos, arranca en `01_dataset_preparation.ipynb`.
- Si queres entender el entrenamiento, segui con `02_model_training.ipynb`.

### `data/`

- `README.md`: explica el dataset, como se descarga y cual es la estructura esperada.
- `CarDD_release/`: copia local del dataset detectada en este workspace.
- `CarDD_release/CarDD_COCO/annotations/`: JSON COCO usados por el notebook como fuente de verdad.
- `CarDD_release/CarDD_SOD/`: otra variante del dataset que hoy no se usa en el pipeline implementado.
- Las imagenes pesadas no deberian versionarse en Git.

### `prod/`

- Carpeta para codigo reutilizable fuera del notebook.
- `detection_dataset.py`: dataset de deteccion, transforms y `collate_fn` reutilizables.
- `__init__.py`: exporta los componentes principales del modulo.
- `detection_models.py`: factory y variantes de Faster R-CNN.
- `detection_training.py`: loops de entrenamiento y checkpoints.
- `detection_metrics.py`: evaluacion con mAP.

## Que va en cada carpeta

### `dev/`

Zona de exploracion, notebooks, prototipos y validacion interactiva. Todo lo que todavia este en etapa experimental o de descubrimiento deberia vivir primero aca.

### `prod/`

Zona pensada para pasar logica estabilizada desde notebooks a modulos Python reutilizables. Si en el futuro se extrae la logica del notebook a scripts o paquetes, este es el lugar mas natural.

### `data/`

Zona del dataset local, manifests y documentacion de datos. No deberia llenarse con artefactos innecesarios ni con imagenes versionadas en Git. La documentacion de datos debe mantenerse aca.

## Fuente de verdad: donde buscar cada cosa

### Si queres entender de que va el proyecto

- Leer `README.md`.
- Leer este archivo.

### Si queres entender el dataset y su estructura

- Leer `data/README.md`.
- Revisar `data/CarDD_release/CarDD_COCO/annotations/*.json`.

### Si queres entender la logica real del pipeline

- Leer `dev/01_dataset_preparation.ipynb`.
- Leer `prod/detection_dataset.py` para la version reutilizable del dataset.
- Leer `dev/02_model_training.ipynb` y `prod/detection_training.py` para la semana 3.

### Si queres saber como se detecta o descarga el dataset

- Buscar en el notebook las funciones `ensure_gdown`, `download_cardd_zip`, `extract_cardd_zip` y `find_dataset_root`.

### Si queres saber como se transforman las anotaciones COCO

- Ir a la seccion `## 4. Preparacion de anotaciones COCO para deteccion` del notebook.
- Ahi se parsean categorias, imagenes y annotations, y se convierten boxes de `XYWH` a `XYXY`.

### Si queres saber como se separan train, val y test

- Ir a la seccion `## 5. Uso de splits oficiales train / val / test`.
- Ahi se construyen `train_records`, `val_records` y `test_records` desde `split_records`.

### Si queres saber como funciona el `Dataset` de PyTorch

- Ir a la clase `CarDamageDetectionDataset` en `prod/detection_dataset.py`.
- El notebook la importa y la usa en la seccion de `DataLoaders`.

### Si queres saber como funcionan las transforms y augmentations

- Ir a `ComposeDetection`, `ToTensorDetection` y `RandomHorizontalFlipDetection` en `prod/detection_dataset.py`.

### Si queres saber como se construyen los `DataLoader`s

- Ir a la seccion `## 9. DataLoaders` del notebook y a `collate_fn` en `prod/detection_dataset.py`.

### Si queres saber como se entrena el modelo

- Ir a `prod/detection_models.py`.
- Ir a `prod/detection_training.py`.
- Ir a `dev/02_model_training.ipynb`.

### Si queres saber como se evalua con mAP

- Ir a `prod/detection_metrics.py`.
- La metrica principal es `mAP@50:95` y la secundaria `mAP@50`.

### Si queres inspeccionar visualmente samples y boxes

- Ir a `draw_boxes`, `show_augmentations` y `show_batch`.

## Flujo del notebook principal

### 1. Imports y configuracion

Define seeds, detecta si corre en local o Colab, crea `DATA_DIR`, `RAW_DIR` y `PROCESSED_DIR`.

### 2. Descarga / ubicacion del dataset

Busca el dataset en varias rutas conocidas. Si no lo encuentra, intenta descargar `CarDD_release.zip` con `gdown` y extraerlo automaticamente.

### 3. Exploracion de estructura

Valida la estructura de carpetas y cuenta imagenes de la parte COCO.

### 4. Preparacion de anotaciones COCO

Lee `instances_train2017.json`, `instances_val2017.json` e `instances_test2017.json`, mapea categorias a indices, agrupa anotaciones por imagen y construye registros por imagen con:

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

Tambien arma dos estructuras auxiliares:

- `detection_df`: resumen tabular para inspeccion
- `csv_manifest_df`: manifiesto liviano por imagen, hoy solo en memoria

### 5. Distribucion de clases y splits

Calcula frecuencias por clase y por split. Tambien expone conteos de imagenes y boxes para train, val y test.

### 6. Definicion de transforms y `Dataset`

Se definen wrappers simples para deteccion con soporte para boxes:

- `ComposeDetection`
- `ToTensorDetection`
- `RandomHorizontalFlipDetection`
- `CarDamageDetectionDataset`

### 7. Construccion de `DataLoader`s

Se crean `train_loader`, `val_loader` y `test_loader` usando `collate_fn` para batches con numero variable de boxes por imagen.

### 8. Verificacion visual y estructural

Se dibujan boxes sobre imagenes transformadas y se inspecciona un batch real para verificar shapes, rangos y consistencia.

## Funciones y clases principales

### Descarga y localizacion

- `ensure_gdown()`: garantiza que `gdown` este instalado.
- `download_cardd_zip(...)`: descarga el ZIP del dataset desde Google Drive.
- `extract_cardd_zip(...)`: extrae el ZIP en `data/`.
- `find_dataset_root()`: busca el dataset en rutas esperadas y devuelve la primera ruta valida.

### Construccion de datos de deteccion

- Bloque de parsing COCO de la seccion 4: convierte JSON COCO en registros consumibles por PyTorch.
- `box_count_table(records)`: resume cantidad de boxes por clase.

### Transforms y dataset

- `ComposeDetection`: encadena transforms que reciben `(image, target)`.
- `ToTensorDetection`: convierte la imagen a tensor y mantiene el target.
- `RandomHorizontalFlipDetection`: aplica flip horizontal y corrige coordenadas de boxes.
- `CarDamageDetectionDataset`: abre imagenes, construye `target` y aplica transforms.

### Data loading y visualizacion

- `collate_fn(batch)`: devuelve listas de imagenes y targets para deteccion.
- `draw_boxes(...)`: dibuja boxes y labels sobre una imagen.
- `show_augmentations(...)`: muestra augmentations aplicadas sobre una misma muestra.
- `show_batch(...)`: visualiza un batch real del `DataLoader`.

## Estructura de datos interna importante

### `record`

La unidad central del pipeline es un diccionario por imagen. Si en el futuro se migra el notebook a scripts, esta estructura debe preservarse o documentarse claramente.

Campos actuales del `record`:

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

El `target` que devuelve el dataset esta alineado con el formato comun de deteccion en PyTorch:

- `boxes`: `torch.float32`
- `labels`: `torch.int64`
- `image_id`: `torch.int64`
- `area`: `torch.float32`
- `iscrowd`: `torch.int64`

## Dependencias

`requirements.txt` hoy declara:

- `torch`
- `torchvision`
- `pandas`
- `numpy`
- `matplotlib`
- `pillow`
- `tqdm`
- `jupyter`
- `kaggle`
- `gdown`
- `torchmetrics`
- `faster-coco-eval`

`kaggle` esta instalado pero no forma parte del flujo principal implementado hoy. `gdown` si se usa activamente para descarga automatica. `torchmetrics` y `faster-coco-eval` se usan para la evaluacion de deteccion con mAP.

## Comandos utiles

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Abrir el notebook principal:

```bash
jupyter notebook dev/01_dataset_preparation.ipynb
```

## Inconsistencias y huecos actuales

- `README.md` y `data/README.md` mencionan CSVs versionados que hoy no estan presentes.
- El notebook construye `csv_manifest_df`, pero no hace `to_csv(...)`.
- `app.py` y `utils.py` existen pero no concentran logica real todavia.
- La logica reutilizable de datos ya vive en `prod/` y el entrenamiento base de semana 3 esta modularizado para Faster R-CNN.
- El dataset local presente en este workspace incluye anotaciones y archivos auxiliares, pero las imagenes no estan versionadas en Git.

## Recomendaciones para evolucionar el repo

- Mantener `dev/01_dataset_preparation.ipynb` como referencia funcional de datos y `dev/02_model_training.ipynb` como referencia funcional de entrenamiento.
- Mantener `prod/` como lugar principal para codigo reutilizable de datos, modelos, entrenamiento y metricas.
- Si se agregan scripts de entrenamiento, separar claramente preparacion de datos, definicion de modelo, entrenamiento, evaluacion e inferencia.

## Regla practica para cualquiera que entre al repo

Si necesitas entender algo rapido:

1. Lee `README.md` para el panorama general.
2. Lee `data/README.md` para el dataset.
3. Lee `dev/01_dataset_preparation.ipynb` para la implementacion real.
4. Asumi que `app.py` y `utils.py` todavia no son la fuente principal de verdad, y que la implementacion reutilizable actual vive en `prod/`.
