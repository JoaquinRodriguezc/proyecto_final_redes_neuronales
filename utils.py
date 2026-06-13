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


def is_detection_test_report_complete(report: dict | None) -> bool:
    if not isinstance(report, dict):
        return False

    required_top_level_keys = {
        "run_id",
        "checkpoint_path",
        "summary",
        "class_metrics",
        "pr_curves",
        "dataset_diagnostics",
        "nms_sensitivity",
    }
    if not required_top_level_keys.issubset(report):
        return False

    summary = report.get("summary") or {}
    required_summary_keys = {"map", "map_50", "map_75", "mar_100"}
    if not required_summary_keys.issubset(summary):
        return False

    return isinstance(report.get("pr_curves"), list) and isinstance(report.get("class_metrics"), list)


def _format_metric_4_decimals(value):
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _build_info_cards_html(summary: dict) -> str:
    cards = [
        ("mAP@50:95", _format_metric_4_decimals(summary.get("map"))),
        ("mAP@50", _format_metric_4_decimals(summary.get("map_50"))),
        ("mAP@75", _format_metric_4_decimals(summary.get("map_75"))),
        ("mAR@100", _format_metric_4_decimals(summary.get("mar_100"))),
    ]
    return "".join(
        f"""
        <article class="metric-card">
            <span class="metric-label">{escape(label)}</span>
            <strong class="metric-value">{escape(value)}</strong>
        </article>
        """
        for label, value in cards
    )


def _build_html_table(data, columns=None, classes="results-table"):
    frame = pd.DataFrame(data)
    if columns is not None and not frame.empty:
        frame = frame[columns]
    if frame.empty:
        return '<p class="empty-state">No hay datos disponibles para esta seccion.</p>'
    return frame.to_html(index=False, classes=classes, border=0)


def _make_pr_curve_svg(curve: dict, width: int = 420, height: int = 260) -> str:
    padding_left = 44
    padding_right = 18
    padding_top = 20
    padding_bottom = 34
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom
    recall_values = curve.get("recall") or []
    precision_values = curve.get("precision") or []

    if not recall_values or not precision_values:
        return '<div class="empty-state">Sin puntos suficientes para la curva PR.</div>'

    def scale_x(value):
        return padding_left + float(value) * plot_width

    def scale_y(value):
        return padding_top + (1.0 - float(value)) * plot_height

    polyline_points = " ".join(
        f"{scale_x(recall_value):.2f},{scale_y(precision_value):.2f}"
        for recall_value, precision_value in zip(recall_values, precision_values)
    )

    tick_labels = []
    grid_lines = []
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = scale_x(tick)
        y = scale_y(tick)
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{padding_top}" x2="{x:.2f}" y2="{padding_top + plot_height}" class="grid-line" />'
        )
        grid_lines.append(
            f'<line x1="{padding_left}" y1="{y:.2f}" x2="{padding_left + plot_width}" y2="{y:.2f}" class="grid-line" />'
        )
        tick_labels.append(
            f'<text x="{x:.2f}" y="{height - 10}" class="axis-label" text-anchor="middle">{tick:.2f}</text>'
        )
        tick_labels.append(
            f'<text x="{padding_left - 10}" y="{y + 4:.2f}" class="axis-label" text-anchor="end">{tick:.2f}</text>'
        )

    return f"""
    <svg class="pr-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Curva precision-recall de {escape(curve.get('class_name', 'clase'))}">
        <rect x="0" y="0" width="{width}" height="{height}" rx="16" ry="16" class="chart-bg" />
        {''.join(grid_lines)}
        <line x1="{padding_left}" y1="{padding_top + plot_height}" x2="{padding_left + plot_width}" y2="{padding_top + plot_height}" class="axis-line" />
        <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + plot_height}" class="axis-line" />
        <polyline points="{polyline_points}" class="pr-line" />
        {''.join(tick_labels)}
        <text x="{padding_left + (plot_width / 2):.2f}" y="{height - 4}" class="axis-title" text-anchor="middle">Recall</text>
        <text x="18" y="{padding_top + (plot_height / 2):.2f}" class="axis-title" text-anchor="middle" transform="rotate(-90 18 {padding_top + (plot_height / 2):.2f})">Precision</text>
    </svg>
    """


def export_detection_test_report_html(
    report: dict,
    output_path,
    title: str = "Reporte final de deteccion en test",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = report.get("summary") or {}
    class_metrics = report.get("class_metrics") or []
    pr_curves = report.get("pr_curves") or []
    dataset_diagnostics = report.get("dataset_diagnostics") or {}
    nms_sensitivity = report.get("nms_sensitivity") or {}

    class_metrics_html = _build_html_table(
        class_metrics,
        columns=["class_id", "class_name", "map_per_class", "mar_100_per_class"],
    )
    dataset_diagnostics_html = _build_html_table(
        dataset_diagnostics.get("per_class") or [],
        columns=[
            "class_id",
            "class_name",
            "annotation_count",
            "image_count",
            "median_bbox_area",
            "median_bbox_area_pct",
            "mean_instances_per_image",
            "max_instances_per_image",
        ],
    )
    nms_results = nms_sensitivity.get("results") or []
    nms_table_rows = [
        {
            "nms_threshold": row.get("nms_threshold"),
            "map": row.get("map"),
            "map_50": row.get("map_50"),
            "map_75": row.get("map_75"),
            "mar_100": row.get("mar_100"),
        }
        for row in nms_results
    ]
    nms_html = _build_html_table(
        nms_table_rows,
        columns=["nms_threshold", "map", "map_50", "map_75", "mar_100"],
    )

    pr_cards_html = "".join(
        f"""
        <article class="pr-card">
            <div class="pr-card-header">
                <h3>{escape(curve.get('class_name', 'Clase'))}</h3>
                <p>AP@50={_format_metric_4_decimals(curve.get('ap_50'))}</p>
            </div>
            {_make_pr_curve_svg(curve)}
        </article>
        """
        for curve in pr_curves
    )

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)}</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f4f7f9;
            --surface: rgba(255, 255, 255, 0.9);
            --surface-strong: #ffffff;
            --border: rgba(17, 39, 54, 0.12);
            --text: #112736;
            --muted: #5b6b77;
            --accent: #1565c0;
            --accent-soft: rgba(21, 101, 192, 0.1);
            --shadow: 0 18px 40px rgba(17, 39, 54, 0.09);
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 32px 20px 48px;
            font-family: Aptos, Manrope, "Segoe UI", sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, #ffffff 0%, rgba(255, 255, 255, 0) 32%),
                linear-gradient(135deg, #eef3f6 0%, #dfe8ed 100%);
        }}

        .page {{
            max-width: 1500px;
            margin: 0 auto;
        }}

        .hero,
        .section-shell {{
            border: 1px solid var(--border);
            border-radius: 20px;
            background: var(--surface);
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
        }}

        .hero {{
            padding: 28px 30px;
            margin-bottom: 20px;
        }}

        .hero h1 {{
            margin: 0 0 8px;
            font-size: clamp(1.9rem, 3vw, 2.8rem);
        }}

        .hero p {{
            margin: 0;
            color: var(--muted);
            max-width: 920px;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-top: 22px;
        }}

        .metric-card {{
            padding: 18px 18px 16px;
            border-radius: 16px;
            background: var(--surface-strong);
            border: 1px solid var(--border);
        }}

        .metric-label {{
            display: block;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
            margin-bottom: 8px;
        }}

        .metric-value {{
            font-size: 1.5rem;
        }}

        .section-shell {{
            margin-bottom: 20px;
            overflow: hidden;
        }}

        .section-header {{
            padding: 20px 24px 10px;
        }}

        .section-header h2 {{
            margin: 0 0 6px;
            font-size: 1.3rem;
        }}

        .section-header p {{
            margin: 0;
            color: var(--muted);
        }}

        .section-body {{
            padding: 0 24px 24px;
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
            padding: 14px 15px;
            text-align: left;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
            background: rgba(240, 245, 249, 0.98);
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}

        .results-table tbody td {{
            padding: 14px 15px;
            border-bottom: 1px solid rgba(17, 39, 54, 0.08);
            vertical-align: top;
        }}

        .results-table tbody tr:nth-child(even) {{
            background: rgba(244, 248, 250, 0.84);
        }}

        .pr-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 18px;
        }}

        .pr-card {{
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--surface-strong);
            padding: 18px;
        }}

        .pr-card-header {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 10px;
        }}

        .pr-card-header h3 {{
            margin: 0;
            font-size: 1.05rem;
        }}

        .pr-card-header p {{
            margin: 0;
            color: var(--muted);
            font-size: 0.92rem;
            white-space: nowrap;
        }}

        .pr-chart {{
            width: 100%;
            height: auto;
        }}

        .chart-bg {{
            fill: #f7fbff;
            stroke: rgba(21, 101, 192, 0.08);
        }}

        .grid-line {{
            stroke: rgba(17, 39, 54, 0.08);
            stroke-width: 1;
        }}

        .axis-line {{
            stroke: rgba(17, 39, 54, 0.25);
            stroke-width: 1.4;
        }}

        .pr-line {{
            fill: none;
            stroke: var(--accent);
            stroke-width: 3;
            stroke-linecap: round;
            stroke-linejoin: round;
        }}

        .axis-label,
        .axis-title {{
            fill: var(--muted);
            font-size: 11px;
        }}

        .empty-state {{
            margin: 0;
            padding: 18px;
            border-radius: 14px;
            background: var(--accent-soft);
            color: var(--muted);
        }}

        .summary-note {{
            margin-top: 14px;
            padding: 14px 16px;
            border-radius: 14px;
            background: var(--accent-soft);
            color: var(--text);
        }}

        @media (max-width: 900px) {{
            body {{
                padding: 18px 12px 28px;
            }}

            .hero,
            .section-shell {{
                border-radius: 16px;
            }}

            .section-header,
            .section-body,
            .hero {{
                padding-left: 18px;
                padding-right: 18px;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <section class="hero">
            <h1>{escape(title)}</h1>
            <p>
                Reporte final de test para {escape(report.get('best_experiment', 'el mejor experimento'))}.
                Resume metricas globales, diferencias por clase, sensibilidad a NMS y curvas precision-recall por clase.
            </p>
            <div class="metric-grid">
                {_build_info_cards_html(summary)}
            </div>
            <div class="summary-note">
                {escape((nms_sensitivity.get('conclusion') or 'Sin conclusion disponible para el barrido de NMS.'))}
            </div>
        </section>

        <section class="section-shell">
            <div class="section-header">
                <h2>Metricas por clase</h2>
                <p>mAP y mAR del checkpoint final sobre el split de test.</p>
            </div>
            <div class="section-body">
                <div class="table-wrap">{class_metrics_html}</div>
            </div>
        </section>

        <section class="section-shell">
            <div class="section-header">
                <h2>Diagnostico del dataset</h2>
                <p>
                    Split: {escape(str(dataset_diagnostics.get('split')))} |
                    imagenes: {dataset_diagnostics.get('num_images', 0)} |
                    anotaciones: {dataset_diagnostics.get('num_annotations', 0)}
                </p>
            </div>
            <div class="section-body">
                <div class="table-wrap">{dataset_diagnostics_html}</div>
            </div>
        </section>

        <section class="section-shell">
            <div class="section-header">
                <h2>Sensibilidad a NMS</h2>
                <p>
                    score_threshold={escape(str(nms_sensitivity.get('score_threshold')))} |
                    detections_per_img={escape(str(nms_sensitivity.get('detections_per_img')))} |
                    baseline_nms={escape(str(nms_sensitivity.get('baseline_nms_threshold')))}
                </p>
            </div>
            <div class="section-body">
                <div class="table-wrap">{nms_html}</div>
            </div>
        </section>

        <section class="section-shell">
            <div class="section-header">
                <h2>Curvas precision-recall por clase</h2>
                <p>Una curva por clase a IoU=0.50, area=all y max_dets=100.</p>
            </div>
            <div class="section-body">
                <div class="pr-grid">
                    {pr_cards_html or '<p class="empty-state">No se generaron curvas precision-recall.</p>'}
                </div>
            </div>
        </section>
    </div>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")
    return output_path


def detection_test_result_paths(run_id: str, output_dir) -> dict:
    output_dir = Path(output_dir)
    safe_run_id = str(run_id).strip()
    return {
        "json": output_dir / f"{safe_run_id}_test_result.json",
        "html": output_dir / f"{safe_run_id}_test_result_report.html",
    }


def save_detection_test_result_artifacts(
    report: dict,
    output_dir,
    title: str = "Reporte final de deteccion en test",
    canonical_json_path=None,
    canonical_html_path=None,
    update_canonical: bool = False,
) -> dict:
    if not report.get("run_id"):
        raise ValueError("El reporte de test debe incluir run_id para persistir artefactos por corrida.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = detection_test_result_paths(report["run_id"], output_dir)

    paths["json"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    export_detection_test_report_html(report, paths["html"], title=title)

    if update_canonical:
        if canonical_json_path is not None:
            canonical_json_path = Path(canonical_json_path)
            canonical_json_path.parent.mkdir(parents=True, exist_ok=True)
            canonical_json_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
                encoding="utf-8",
            )
        if canonical_html_path is not None:
            export_detection_test_report_html(report, canonical_html_path, title=title)

    return paths


def archive_canonical_detection_test_result(
    canonical_json_path,
    output_dir,
    overwrite: bool = False,
) -> Path | None:
    canonical_json_path = Path(canonical_json_path)
    if not canonical_json_path.exists():
        return None

    report = json.loads(canonical_json_path.read_text(encoding="utf-8"))
    run_id = report.get("run_id")
    if not run_id:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archived_path = detection_test_result_paths(run_id, output_dir)["json"]
    if archived_path.exists() and not overwrite:
        return archived_path

    archived_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return archived_path


def load_detection_test_results(test_results_dir) -> list[dict]:
    test_results_dir = Path(test_results_dir)
    if not test_results_dir.exists():
        return []

    reports = []
    for result_path in sorted(test_results_dir.glob("*_test_result.json")):
        report = json.loads(result_path.read_text(encoding="utf-8"))
        report["_result_json_path"] = result_path.as_posix()
        report["_result_html_path"] = detection_test_result_paths(
            report.get("run_id", result_path.stem.replace("_test_result", "")),
            test_results_dir,
        )["html"].as_posix()
        reports.append(report)
    return reports


def build_detection_test_results_comparison_df(
    runs_manifest_path,
    test_results_dir,
) -> pd.DataFrame:
    run_records = {
        record.get("run_id"): record
        for record in load_jsonl_records(runs_manifest_path)
        if record.get("run_id")
    }
    rows = []
    for report in load_detection_test_results(test_results_dir):
        run_id = report.get("run_id")
        run_record = run_records.get(run_id, {})
        class_metrics = {
            row.get("class_name"): row
            for row in report.get("class_metrics", [])
        }
        rows.append(
            {
                "run_id": run_id,
                "experiment": report.get("best_experiment") or run_record.get("name"),
                "checkpoint_path": report.get("checkpoint_path") or run_record.get("checkpoint_path"),
                "test_map": report.get("test_map"),
                "test_map_50": report.get("test_map_50"),
                "dent_map": (class_metrics.get("dent") or {}).get("map_per_class"),
                "scratch_map": (class_metrics.get("scratch") or {}).get("map_per_class"),
                "crack_map": (class_metrics.get("crack") or {}).get("map_per_class"),
                "result_json": report.get("_result_json_path"),
                "result_html": report.get("_result_html_path"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "experiment",
                "checkpoint_path",
                "test_map",
                "test_map_50",
                "dent_map",
                "scratch_map",
                "crack_map",
                "result_json",
                "result_html",
            ]
        )

    return pd.DataFrame(rows).sort_values(by="test_map", ascending=False, na_position="last").reset_index(drop=True)
