from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from utils import export_results_comparison_html


class ExportResultsComparisonHtmlTest(unittest.TestCase):
    def test_export_results_comparison_html_generates_expected_table(self):
        # Simula un resumen mínimo de corridas para ejercitar la exportación HTML.
        results_df = pd.DataFrame(
            [
                {
                    "name": "fasterrcnn_head_only",
                    "model_name": "fasterrcnn",
                    "best_epoch": 3,
                    "best_map": 0.41,
                    "best_map_50": 0.67,
                    "best_val_loss": 1.23,
                },
                {
                    "name": "retinanet_partial_backbone_fixed_size",
                    "model_name": "retinanet",
                    "best_epoch": 4,
                    "best_map": 0.38,
                    "best_map_50": 0.61,
                    "best_val_loss": 1.35,
                },
            ]
        )

        output_path = Path(__file__).resolve().parent / "artifacts" / "results_comparison.html"

        # Genera el reporte visual que luego consumen los notebooks.
        generated_path = export_results_comparison_html(
            results_df,
            output_path,
            title="Comparación mock de resultados",
        )

        self.assertEqual(generated_path, output_path)
        self.assertTrue(output_path.exists())

        # Verifica que el archivo tenga la estructura básica esperada.
        html_content = output_path.read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn("<table", html_content)
        self.assertIn("Comparación mock de resultados", html_content)
        self.assertIn("fasterrcnn_head_only", html_content)
        self.assertIn("retinanet", html_content)
        self.assertIn("mAP", html_content)


if __name__ == "__main__":
    unittest.main()
