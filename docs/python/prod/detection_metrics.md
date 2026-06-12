# `prod/detection_metrics.py`

## Proposito del archivo

Este modulo concentra la evaluacion de deteccion basada en `torchmetrics`.

Su objetivo principal es calcular `mAP@50:95`, `mAP@50` y metricas relacionadas de manera reutilizable fuera del notebook.

## Funciones

### `create_map_metric(class_metrics: bool = True)`

Crea una instancia de `MeanAveragePrecision`.

Configuracion principal:

- `box_format="xyxy"`
- `iou_type="bbox"`
- `class_metrics=class_metrics`

Primero intenta usar el backend `faster_coco_eval`. Si la version instalada de `torchmetrics` no acepta ese argumento, hace fallback al constructor sin `backend`.

### `_move_dict_to_cpu(d: dict) -> dict`

Helper interno que mueve a CPU todos los tensores de un diccionario.

Se usa porque `torchmetrics` espera recibir predicciones y targets en CPU al acumular batches.

### `summarize_map_results(results: dict) -> dict`

Convierte el resultado de `torchmetrics` a tipos Python mas faciles de serializar o imprimir.

Comportamiento:

- tensores escalares -> `float`
- tensores no escalares -> `list`
- otros valores -> se devuelven tal cual

### `evaluate_map(model, dataloader, device, class_metrics: bool = True, max_batches=None)`

Calcula mAP recorriendo un dataloader completo o un subconjunto de batches.

Flujo:

1. crea el acumulador de metricas
2. guarda el modo actual del modelo
3. pone el modelo en evaluacion
4. desactiva gradientes
5. mueve imagenes al dispositivo
6. obtiene predicciones del modelo
7. mueve predicciones y targets a CPU
8. actualiza la metrica batch a batch
9. restaura el modo original del modelo
10. computa y serializa el resultado final

Devuelve un diccionario con metricas agregadas de deteccion.

### `extract_main_map_metrics(results)`

Extrae del resultado completo un subconjunto de metricas principales:

- `map`
- `map_50`
- `map_75`
- `mar_100`

Es util para guardar historiales compactos por epoca.

## Como se usa en el proyecto

`run_detection_experiment(...)` en `prod/detection_training.py` usa este modulo para evaluar el conjunto de validacion al final de cada epoca y decidir el mejor checkpoint.
