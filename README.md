# Proyecto Final — Detección/Clasificación de daños en autos con CarDD

Repositorio base para preparar un pipeline reproducible en PyTorch usando **CarDD (Car Damage Detection Dataset)**. En esta etapa el repo cubre preparación del dataset para detección y una base de entrenamiento con fine-tuning de **Faster R-CNN**: exploración, lectura de COCO, `Dataset` custom, transforms, `DataLoader`s, módulos reutilizables en `prod/` y notebooks de verificación y entrenamiento.

## Dataset usado

- **Dataset:** CarDD: Car Damage Detection Dataset
- **Fuente oficial:** https://cardd-ustc.github.io/
- **Formato usado en esta entrega:** `CarDD_COCO`
- **ZIP de descarga directa:** `https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view`
- **Nota:** CarDD fue publicado para tareas de detección/segmentación de daños. En este repo se usa COCO como fuente principal para detección usando directamente los JSON COCO como fuente de verdad.

## Estructura del repo

```text
.
├── data/
│   ├── README.md
│   └── CarDD_release/
├── dev/
│   ├── 01_dataset_preparation.ipynb
│   └── 02_model_training.ipynb
├── prod/
│   ├── __init__.py
│   ├── detection_dataset.py
│   ├── detection_models.py
│   ├── detection_training.py
│   └── detection_metrics.py
├── requirements.txt
├── .gitignore
├── CLAUDE.md
├── AGENTS.md
└── README.md
```

## Reproducibilidad local

1. Clonar el repositorio.
2. Crear entorno virtual.
3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Abrir `dev/01_dataset_preparation.ipynb`.
5. Ejecutar el notebook. Si el dataset no está presente localmente, intentará descargarlo y extraerlo automáticamente.

## Google Colab

1. Subir `dev/01_dataset_preparation.ipynb` a Colab o abrirlo desde GitHub.
2. Ejecutar la celda **Instalación opcional de dependencias**.
3. Ejecutar el resto del notebook.
4. El dataset se descargará automáticamente en `/content/data/` si no existe.

## Notas importantes

- Las imágenes del dataset **no** se suben a GitHub.
- Se versionan archivos livianos como notebooks, módulos Python y documentación.
- El notebook arma un manifiesto en memoria a partir de los splits oficiales de CarDD COCO, pero hoy no persiste `data/train.csv`, `data/val.csv` y `data/test.csv` a disco.
- La base de semana 3 compara tres variantes de `Faster R-CNN` y usa `mAP@50:95` y `mAP@50` como métricas principales para detección.
