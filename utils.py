from __future__ import annotations

import json
from html import escape
from pathlib import Path, PureWindowsPath
from datetime import timedelta

import pandas as pd


# Permite serializar valores frecuentes del proyecto al guardar JSON.
def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


# Agrega una línea JSON al manifest sin pisar corridas anteriores.
def append_jsonl_record(path, record: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, default=_json_default)
        f.write("\n")
    return path


# Lee un archivo JSONL y devuelve una lista de diccionarios.
def load_jsonl_records(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Convierte una ruta absoluta a un formato portable relativo al repo cuando es posible.
def to_portable_path(path, base_dir=None) -> str:
    path = Path(path)
    if base_dir is not None:
        base_path = Path(base_dir).resolve()
        try:
            return path.resolve().relative_to(base_path).as_posix()
        except ValueError:
            pass
    if not path.is_absolute():
        return path.as_posix()
    return str(path)


# Resuelve rutas históricas de Windows o rutas relativas guardadas en manifests.
def resolve_portable_path(path, base_dir=None, fallback_dir=None) -> Path:
    raw_path = str(path)
    candidate = Path(raw_path)
    portable_candidate = Path(PureWindowsPath(raw_path).as_posix())

    # Evalúa variantes POSIX y Windows para reutilizar manifests viejos.
    candidates = []

    for item in (candidate, portable_candidate):
        if item not in candidates:
            candidates.append(item)

    if base_dir is not None:
        base_path = Path(base_dir)
        for item in (candidate, portable_candidate):
            combined = base_path / item
            if combined not in candidates:
                candidates.append(combined)

    looks_like_windows_path = (
        "\\" in raw_path
        or (len(raw_path) >= 2 and raw_path[1] == ":")
    )
    if fallback_dir is not None and looks_like_windows_path:
        filename_candidate = Path(fallback_dir) / PureWindowsPath(raw_path).name
        if filename_candidate not in candidates:
            candidates.append(filename_candidate)

    # Devuelve la primera ruta candidata que realmente exista en disco.
    for item in candidates:
        if item.exists():
            return item.resolve()

    tried_paths = ", ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"Checkpoint path not found: {raw_path}. Tried: {tried_paths}")


# Construye el registro serializable de una corrida a partir del resultado del entrenamiento.
def make_experiment_run_record(experiment: dict, run_result: dict, best_row: dict) -> dict:
    created_at = run_result["training_end_time"]
    optimizer_name = str(experiment.get("optimizer_name", "unknown")).strip().lower()
    run_id = (
        f"{created_at.replace(':', '').replace('-', '').replace('T', '_')}_"
        f"{experiment['name']}_{optimizer_name}"
    )
    return {
        "run_id": run_id,
        "created_at": created_at,
        "name": experiment["name"],
        "optimizer_name": optimizer_name,
        "trainable_backbone_layers": experiment.get("trainable_backbone_layers"),
        "num_epochs": experiment.get("num_epochs"),
        "config": experiment,
        "best_epoch": run_result["best_epoch"],
        "best_map": best_row.get("map"),
        "best_map_50": best_row.get("map_50"),
        "best_map_75": best_row.get("map_75"),
        "best_val_loss": best_row.get("val_loss"),
        "checkpoint_path": run_result["best_checkpoint_path"],
        "training_start_time": run_result["training_start_time"],
        "training_end_time": run_result["training_end_time"],
        "training_duration_seconds": run_result["training_duration_seconds"],
        "history": run_result["history"],
    }


# Carga el manifest histórico y normaliza columnas útiles para análisis en pandas.
def load_experiment_runs(path) -> pd.DataFrame:
    records = load_jsonl_records(path)
    if not records:
        return pd.DataFrame()

    normalized_records = []
    for record in records:
        normalized_record = dict(record)
        config = normalized_record.get("config") or {}
        normalized_record["optimizer_name"] = normalized_record.get(
            "optimizer_name",
            config.get("optimizer_name", "unknown"),
        )
        normalized_record["trainable_backbone_layers"] = normalized_record.get(
            "trainable_backbone_layers",
            config.get("trainable_backbone_layers"),
        )
        normalized_record["num_epochs"] = normalized_record.get(
            "num_epochs",
            config.get("num_epochs"),
        )
        normalized_records.append(normalized_record)

    return pd.DataFrame(normalized_records)


# Formatea duración en segundos como HH:MM:SS para tablas comparativas.
def _format_duration_hms(total_seconds) -> str:
    if pd.isna(total_seconds):
        return ""
    rounded_seconds = int(round(float(total_seconds)))
    return str(timedelta(seconds=rounded_seconds))


# Redondea métricas numéricas para mostrarlas de forma compacta.
def _format_metric_2_decimals(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


# Exporta una tabla HTML estilizada con el resumen de corridas registradas.
def export_results_comparison_html(
    results_df: pd.DataFrame,
    output_path,
    title: str = "Comparacion de resultados",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Trabaja sobre una copia para no mutar el DataFrame original del notebook.
    display_df = results_df.copy()
    if "training_duration_seconds" in display_df.columns:
        display_df["duration_hms"] = display_df["training_duration_seconds"].apply(_format_duration_hms)
    for metric_column in ["best_map", "best_map_50", "best_map_75", "best_val_loss"]:
        if metric_column in display_df.columns:
            display_df[metric_column] = display_df[metric_column].apply(_format_metric_2_decimals)

    # Prioriza las columnas que mejor explican la comparación experimental.
    preferred_columns = [
        "name",
        "duration_hms",
        "optimizer_name",
        "trainable_backbone_layers",
        "num_epochs",
        "best_epoch",
        "best_map",
        "best_map_50",
        "best_map_75",
        "best_val_loss",
        "checkpoint_path",
    ]
    available_columns = [column for column in preferred_columns if column in display_df.columns]
    if available_columns:
        display_df = display_df[available_columns]

    # Renombra encabezados para la versión visual del reporte.
    display_df = display_df.rename(
        columns={
            "name": "Experimento",
            "duration_hms": "Duracion",
            "optimizer_name": "Optimizer",
            "trainable_backbone_layers": "Capas backbone",
            "num_epochs": "Epocas",
            "best_epoch": "Mejor epoca",
            "best_map": "mAP",
            "best_map_50": "mAP@50",
            "best_map_75": "mAP@75",
            "best_val_loss": "Val loss",
            "checkpoint_path": "Checkpoint",
        }
    )

    table_html = display_df.to_html(index=False, classes="results-table", border=0)

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)}</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f3f6f8;
            --bg-accent: #e8eff2;
            --surface: rgba(255, 255, 255, 0.88);
            --surface-strong: #ffffff;
            --border: rgba(20, 48, 64, 0.12);
            --text: #13212b;
            --muted: #5a6c78;
            --accent: #0f8b8d;
            --shadow: 0 22px 50px rgba(24, 48, 63, 0.10);
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 40px 24px 56px;
            font-family: Aptos, Manrope, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, #ffffff 0%, rgba(255, 255, 255, 0) 34%),
                linear-gradient(135deg, var(--bg) 0%, var(--bg-accent) 100%);
        }}

        .page {{
            max-width: 1500px;
            margin: 0 auto;
        }}

        .hero {{
            margin-bottom: 20px;
            padding: 28px 30px;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(255, 255, 255, 0.82));
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }}

        h1 {{
            margin: 0 0 10px;
            font-size: clamp(1.9rem, 3vw, 2.8rem);
            line-height: 1.05;
            letter-spacing: 0;
        }}

        .subtitle {{
            margin: 0;
            color: var(--muted);
            font-size: 0.98rem;
        }}

        .table-shell {{
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--surface);
            box-shadow: var(--shadow);
            overflow: hidden;
            backdrop-filter: blur(18px);
        }}

        .table-wrap {{
            overflow: auto;
        }}

        table.results-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: var(--surface-strong);
        }}

        .results-table thead th {{
            position: sticky;
            top: 0;
            z-index: 1;
            padding: 14px 16px;
            text-align: left;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--muted);
            background: rgba(243, 248, 250, 0.96);
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}

        .results-table tbody td {{
            padding: 14px 16px;
            font-size: 0.95rem;
            border-bottom: 1px solid rgba(20, 48, 64, 0.08);
            vertical-align: top;
        }}

        .results-table tbody tr:nth-child(even) {{
            background: rgba(244, 249, 250, 0.72);
        }}

        .results-table tbody tr:hover {{
            background: rgba(15, 139, 141, 0.08);
        }}

        .results-table tbody tr:last-child td {{
            border-bottom: 0;
        }}

        .results-table td:nth-child(1),
        .results-table td:nth-child(2),
        .results-table td:nth-child(3),
        .results-table td:nth-child(4),
        .results-table td:nth-child(5),
        .results-table td:nth-child(11) {{
            font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
            font-size: 0.88rem;
        }}

        .results-table td:nth-child(3) {{
            color: var(--accent);
            font-weight: 700;
        }}

        .results-table td:nth-child(2),
        .results-table td:nth-child(6),
        .results-table td:nth-child(7),
        .results-table td:nth-child(8) {{
            white-space: nowrap;
        }}

        .results-table td:nth-child(11) {{
            min-width: 360px;
            color: var(--muted);
            word-break: break-all;
        }}

        .footer-note {{
            padding: 14px 18px 18px;
            color: var(--muted);
            font-size: 0.9rem;
            border-top: 1px solid var(--border);
            background: linear-gradient(180deg, rgba(250, 252, 253, 0.86), rgba(244, 248, 249, 0.95));
        }}

        @media (max-width: 900px) {{
            body {{
                padding: 20px 14px 32px;
            }}

            .hero,
            .table-shell {{
                border-radius: 14px;
            }}

            .results-table thead th,
            .results-table tbody td {{
                padding: 12px 13px;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <section class="hero">
            <h1>{escape(title)}</h1>
            <p class="subtitle">{len(results_df)} corrida(s) registradas en la comparacion actual.</p>
        </section>
        <section class="table-shell">
            <div class="table-wrap">
                {table_html}
            </div>
            <div class="footer-note">
                Tabla generada automaticamente a partir del manifest de corridas.
            </div>
        </section>
    </div>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")
    return output_path
