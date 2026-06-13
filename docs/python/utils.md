# `utils.py`

## Proposito del archivo

`utils.py` concentra helpers compartidos por los notebooks cuando la logica no pertenece a un modulo especifico de `prod/`.

## Estado actual

- Exporta tablas de comparacion de resultados a HTML.
- Guarda y lee corridas de experimentos en formato JSONL.
- Arma registros serializables a partir de los resultados de entrenamiento.
- Participa en `dev/02_model_training.ipynb` y `dev/03_model_selection.ipynb`.

## Helpers disponibles

### `append_jsonl_record(path, record)`

Agrega un diccionario como una linea JSON a un archivo `.jsonl`.

Se usa para registrar cada experimento terminado sin pisar corridas anteriores.

### `load_jsonl_records(path)`

Lee un archivo `.jsonl` y devuelve una lista de diccionarios.

Si el archivo no existe, devuelve una lista vacia.

### `make_experiment_run_record(experiment, run_result, best_row)`

Construye el registro persistente de una corrida.

El `run_id` se arma con timestamp, nombre del experimento y `optimizer_name` para distinguir mejor corridas parecidas.

Incluye:

- `run_id`
- `created_at`
- `name`
- `optimizer_name`
- `trainable_backbone_layers`
- `num_epochs`
- `config`
- metricas principales de validacion
- `checkpoint_path`
- tiempos de entrenamiento
- `history`

No incluye `best_payload`, porque contiene tensores y pesos no serializables.

### `load_experiment_runs(path)`

Carga las corridas JSONL y las devuelve como `pandas.DataFrame`.

Si una corrida no trae `optimizer_name` como campo de primer nivel, lo reconstruye desde `config` para mantener compatibilidad con manifests anteriores.

Hace lo mismo con `trainable_backbone_layers` y `num_epochs` cuando esos campos todavia no existen como columnas propias.

### `export_results_comparison_html(results_df, output_path, title="Comparacion de resultados")`

Exporta un `DataFrame` a una tabla HTML estilizada, con contenedor responsive, encabezado sticky y formato visual mas legible para comparar corridas.

Cuando existe `training_duration_seconds`, la exportacion agrega una columna visible `duration_hms` en formato `HH:MM:SS` y prioriza esa vista compacta para la tabla comparativa.

## Relacion con el proyecto actual

La logica reusable de dataset, modelos, entrenamiento y metricas sigue en `prod/`.

`utils.py` queda reservado para helpers transversales de notebooks, especialmente persistencia y reportes livianos.
