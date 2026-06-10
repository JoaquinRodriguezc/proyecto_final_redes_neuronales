from __future__ import annotations

from typing import Any

import torch


def _require_torchmetrics():
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError as exc:
        raise ImportError(
            "Para calcular mAP instala torchmetrics y un backend COCO compatible. "
            "Revisa requirements.txt."
        ) from exc

    return MeanAveragePrecision


def create_map_metric(class_metrics: bool = True):
    MeanAveragePrecision = _require_torchmetrics()

    try:
        return MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=class_metrics,
            backend="faster_coco_eval",
        )
    except TypeError:
        return MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=class_metrics,
        )


def _move_prediction_to_cpu(prediction):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in prediction.items()
    }


def _move_target_to_cpu(target):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in target.items()
    }


def _detach_metric_value(value: Any):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    return value


def summarize_map_results(results):
    return {key: _detach_metric_value(value) for key, value in results.items()}


def evaluate_map(
    model,
    dataloader,
    device,
    class_metrics: bool = True,
    max_batches=None,
):
    metric = create_map_metric(class_metrics=class_metrics)
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = [image.to(device) for image in images]
            predictions = model(images)

            predictions_cpu = [_move_prediction_to_cpu(prediction) for prediction in predictions]
            targets_cpu = [_move_target_to_cpu(target) for target in targets]
            metric.update(predictions_cpu, targets_cpu)

    if was_training:
        model.train()

    return summarize_map_results(metric.compute())


def extract_main_map_metrics(results):
    return {
        "map": results.get("map"),
        "map_50": results.get("map_50"),
        "map_75": results.get("map_75"),
        "mar_100": results.get("mar_100"),
    }
