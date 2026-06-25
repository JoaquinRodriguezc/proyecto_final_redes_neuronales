from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def export_results_comparison_html(
    results_df: pd.DataFrame,
    output_path,
    title: str = "Comparación de resultados",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)}</title>
</head>
<body>
    <h1>{escape(title)}</h1>
    {results_df.to_html(index=False)}
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")
    return output_path
