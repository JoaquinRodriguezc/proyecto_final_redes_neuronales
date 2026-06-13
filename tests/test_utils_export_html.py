from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from utils import (
    archive_canonical_detection_test_result,
    build_detection_test_results_comparison_df,
    detection_test_result_paths,
    export_detection_test_report_html,
    export_results_comparison_html,
    is_detection_test_report_complete,
    save_detection_test_result_artifacts,
)


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

    def test_export_detection_test_report_html_renders_sections_and_pr_charts(self):
        report = {
            "run_id": "run_001",
            "best_experiment": "fasterrcnn_mobilenet_v3_large_partial_backbone",
            "checkpoint_path": "dev/experiments/model_best.pth",
            "summary": {
                "map": 0.4466,
                "map_50": 0.6479,
                "map_75": 0.4777,
                "mar_100": 0.5713,
            },
            "class_metrics": [
                {"class_id": 4, "class_name": "glass shatter", "map_per_class": 0.8056, "mar_100_per_class": 0.8380},
                {"class_id": 3, "class_name": "crack", "map_per_class": 0.0866, "mar_100_per_class": 0.2986},
            ],
            "pr_curves": [
                {
                    "class_id": 4,
                    "class_name": "glass shatter",
                    "iou": 0.5,
                    "area": "all",
                    "max_dets": 100,
                    "recall": [0.0, 0.5, 1.0],
                    "precision": [1.0, 0.95, 0.9],
                    "ap_50": 0.95,
                },
                {
                    "class_id": 3,
                    "class_name": "crack",
                    "iou": 0.5,
                    "area": "all",
                    "max_dets": 100,
                    "recall": [0.0, 0.5, 1.0],
                    "precision": [0.4, 0.3, 0.2],
                    "ap_50": 0.3,
                },
            ],
            "dataset_diagnostics": {
                "split": "test",
                "num_images": 374,
                "num_annotations": 785,
                "per_class": [
                    {
                        "class_id": 4,
                        "class_name": "glass shatter",
                        "annotation_count": 71,
                        "image_count": 71,
                        "median_bbox_area": 375222.0,
                        "median_bbox_area_pct": 56.255,
                        "mean_instances_per_image": 1.0,
                        "max_instances_per_image": 1,
                    }
                ],
            },
            "nms_sensitivity": {
                "supported": True,
                "baseline_nms_threshold": 0.5,
                "score_threshold": 0.05,
                "detections_per_img": 100,
                "conclusion": "El barrido de NMS apenas mueve el mAP global.",
                "results": [
                    {"nms_threshold": 0.3, "map": 0.4609, "map_50": 0.6725, "map_75": 0.5130, "mar_100": 0.57},
                    {"nms_threshold": 0.5, "map": 0.4620, "map_50": 0.6810, "map_75": 0.5106, "mar_100": 0.58},
                ],
            },
        }

        output_path = Path(__file__).resolve().parent / "artifacts" / "best_test_result_report.html"

        generated_path = export_detection_test_report_html(
            report,
            output_path,
            title="Reporte final mock",
        )

        self.assertEqual(generated_path, output_path)
        self.assertTrue(output_path.exists())

        html_content = output_path.read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn("Reporte final mock", html_content)
        self.assertIn("Sensibilidad a NMS", html_content)
        self.assertIn("Curvas precision-recall por clase", html_content)
        self.assertIn("glass shatter", html_content)
        self.assertIn("crack", html_content)
        self.assertIn("pr-chart", html_content)

    def test_is_detection_test_report_complete_detects_missing_sections(self):
        complete_report = {
            "run_id": "run_001",
            "checkpoint_path": "dev/experiments/model_best.pth",
            "summary": {"map": 0.1, "map_50": 0.2, "map_75": 0.3, "mar_100": 0.4},
            "class_metrics": [],
            "pr_curves": [],
            "dataset_diagnostics": {},
            "nms_sensitivity": {},
        }
        incomplete_report = {
            "run_id": "run_001",
            "checkpoint_path": "dev/experiments/model_best.pth",
            "summary": {"map": 0.1},
            "class_metrics": [],
        }

        self.assertTrue(is_detection_test_report_complete(complete_report))
        self.assertFalse(is_detection_test_report_complete(incomplete_report))

    def test_save_detection_test_result_artifacts_preserves_canonical_by_default(self):
        output_dir = Path(__file__).resolve().parent / "artifacts" / "test_results"
        canonical_path = Path(__file__).resolve().parent / "artifacts" / "canonical_best_test_result.json"
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_text('{"run_id": "baseline"}', encoding="utf-8")
        report = {
            "run_id": "new_run",
            "best_experiment": "oversample_crop",
            "checkpoint_path": "dev/experiments/new_run_best.pth",
            "test_map": 0.5,
            "test_map_50": 0.7,
            "summary": {"map": 0.5, "map_50": 0.7, "map_75": 0.4, "mar_100": 0.6},
            "class_metrics": [
                {"class_id": 1, "class_name": "dent", "map_per_class": 0.3, "mar_100_per_class": 0.5},
                {"class_id": 2, "class_name": "scratch", "map_per_class": 0.31, "mar_100_per_class": 0.52},
                {"class_id": 3, "class_name": "crack", "map_per_class": 0.09, "mar_100_per_class": 0.3},
            ],
            "pr_curves": [],
            "dataset_diagnostics": {"split": "test", "num_images": 1, "num_annotations": 1, "per_class": []},
            "nms_sensitivity": {"results": [], "conclusion": "mock"},
        }

        paths = save_detection_test_result_artifacts(
            report,
            output_dir,
            canonical_json_path=canonical_path,
            update_canonical=False,
        )

        self.assertEqual(paths, detection_test_result_paths("new_run", output_dir))
        self.assertTrue(paths["json"].exists())
        self.assertTrue(paths["html"].exists())
        self.assertEqual(canonical_path.read_text(encoding="utf-8"), '{"run_id": "baseline"}')

    def test_archive_and_compare_detection_test_results(self):
        artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        output_dir = artifacts_dir / "comparison_test_results"
        manifest_path = artifacts_dir / "comparison_runs_manifest.jsonl"
        canonical_path = artifacts_dir / "comparison_best_test_result.json"
        canonical_report = {
            "run_id": "baseline_run",
            "best_experiment": "baseline",
            "checkpoint_path": "dev/experiments/baseline_best.pth",
            "test_map": 0.44,
            "test_map_50": 0.64,
            "summary": {"map": 0.44, "map_50": 0.64, "map_75": 0.47, "mar_100": 0.57},
            "class_metrics": [
                {"class_name": "dent", "map_per_class": 0.24},
                {"class_name": "scratch", "map_per_class": 0.23},
                {"class_name": "crack", "map_per_class": 0.08},
            ],
            "pr_curves": [],
            "dataset_diagnostics": {},
            "nms_sensitivity": {},
        }
        canonical_path.write_text(json.dumps(canonical_report), encoding="utf-8")
        manifest_path.write_text(
            json.dumps({"run_id": "baseline_run", "name": "baseline"}) + "\n",
            encoding="utf-8",
        )

        archived_path = archive_canonical_detection_test_result(canonical_path, output_dir)
        comparison_df = build_detection_test_results_comparison_df(manifest_path, output_dir)

        self.assertTrue(archived_path.exists())
        self.assertEqual(len(comparison_df), 1)
        self.assertEqual(comparison_df.iloc[0]["run_id"], "baseline_run")
        self.assertEqual(comparison_df.iloc[0]["dent_map"], 0.24)
        self.assertIn("result_html", comparison_df.columns)


if __name__ == "__main__":
    unittest.main()
