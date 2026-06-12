# Dataset

Este proyecto utiliza el dataset **CarDD: Car Damage Detection Dataset**.

Fuente original: https://cardd-ustc.github.io/

ZIP de descarga directa usado en este repo:

`https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view`

CarDD contiene imagenes de vehiculos con danos y anotaciones para tareas de deteccion y segmentacion de danos. En esta entrega se utiliza especificamente la variante **CarDD_COCO**.

## Descarga

El notebook `dev/01_dataset_preparation.ipynb` y el modulo `prod/detection_dataset.py` incluyen funciones para:

- descargar `CarDD_release.zip`
- extraerlo en `data/`
- detectar automaticamente la estructura resultante

Ubicaciones validas detectadas por el proyecto:

- `data/raw/CarDD/`
- `data/CarDD/`
- `data/CarDD_release/`
- `data/CarDD_release/CarDD_COCO/`
- `data/CarDD/CarDD_COCO/`

La estructura esperada para la variante COCO es:

```text
data/CarDD_release/CarDD_COCO/
|-- annotations/
|   |-- instances_train2017.json
|   |-- instances_val2017.json
|   `-- instances_test2017.json
|-- train2017/
|-- val2017/
`-- test2017/
```

## Archivos versionados

- `data/README.md`
- anotaciones COCO livianas, si estan presentes en el workspace

## Estado actual de los CSVs

En el estado actual del repo:

- el notebook construye `csv_manifest_df` en memoria
- no se persisten `data/train.csv`, `data/val.csv` ni `data/test.csv` a disco
- si en el futuro se agregan esos archivos, la documentacion debe actualizarse junto con el codigo

## Reproduccion de la estructura

1. Ejecutar `dev/01_dataset_preparation.ipynb`.
2. Si el dataset no existe localmente, el notebook descargara `CarDD_release.zip` y lo extraera automaticamente en `data/`.
3. Ejecutar el resto del notebook para reconstruir los registros y manifiestos en memoria respetando los splits oficiales de CarDD COCO.

## Google Colab

En Colab, el notebook usa automaticamente `DATA_DIR = /content/data` y descarga/extraccion automatica si el dataset no esta presente.

## Referencias utiles

- `README.md`: setup del repo y contexto general
- `CLAUDE.md`: mapa tecnico del proyecto
- `docs/setup_windows_gpu.md`: guia de entorno local en Windows
- `docs/python/prod/detection_dataset.md`: detalle del modulo reusable de datos
