# Dataset

Este proyecto utiliza el dataset **CarDD: Car Damage Detection Dataset**.

Fuente original: https://cardd-ustc.github.io/

ZIP de descarga directa usado en este repo:

`https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view`

CarDD contiene imágenes de vehículos con daños y anotaciones para tareas de detección y segmentación de daños. En esta entrega se utiliza específicamente la variante **CarDD_COCO**.

## Descarga

El notebook `dev/01_dataset_preparation.ipynb` incluye funciones para:

- descargar `CarDD_release.zip`
- extraerlo en `data/`
- detectar automáticamente la estructura resultante

Ubicaciones válidas detectadas por el notebook:

- `data/raw/CarDD/`
- `data/CarDD/`
- `data/CarDD_release/`
- `data/CarDD_release/CarDD_COCO/`
- `data/CarDD/CarDD_COCO/`

La estructura esperada para la variante COCO es:

```text
data/CarDD_release/CarDD_COCO/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── instances_test2017.json
├── train2017/
├── val2017/
└── test2017/
```

Las imágenes no se versionan en GitHub. Solo se versionan los CSV que reflejan los splits oficiales del dataset:

- `train.csv`
- `val.csv`
- `test.csv`

## Archivos versionados

- `data/README.md`
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

## Reproducción de la estructura

1. Ejecutar `dev/01_dataset_preparation.ipynb`.
2. Si el dataset no existe localmente, el notebook descargará `CarDD_release.zip` y lo extraerá automáticamente en `data/`.
3. Ejecutar el resto del notebook para reconstruir los CSVs respetando los splits oficiales de CarDD COCO.


## Google Colab

En Colab, el notebook usa automáticamente `DATA_DIR = /content/data` y descarga/extracción automática si el dataset no está presente.
