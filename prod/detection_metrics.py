from __future__ import annotations
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def create_map_metric(class_metrics: bool = True):
    try:
        return MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                    class_metrics=class_metrics, backend="faster_coco_eval")
    except TypeError:
        return MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                    class_metrics=class_metrics)

def _move_dict_to_cpu(d: dict) -> dict:
    return {
        k: v.detach().cpu() if torch.is_tensor(v) else v
        for k, v in d.items()
    }


def summarize_map_results(results: dict) -> dict:
    def _detach(value):
        if torch.is_tensor(value):
            return float(value.item()) if value.numel() == 1 else value.detach().cpu().tolist()
        return value
    return {k: _detach(v) for k, v in results.items()}


def evaluate_map(
    model,
    dataloader,
    device,
    class_metrics: bool = True,
    max_batches=None,
):
    # Acumulador de métricas — recibe batches, calcula mAP al final
    metric = create_map_metric(class_metrics=class_metrics)

    # Guarda el modo actual y pone el modelo en evaluación (desactiva Dropout, etc.)
    was_training = model.training
    model.eval()

    # Sin grafo de gradientes — no se necesitan en inferencia
    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(dataloader):

            # Corte anticipado si se pasó un límite de batches
            if max_batches is not None and batch_index >= max_batches:
                break

            # Mueve imágenes al dispositivo (GPU/CPU)
            images = [image.to(device) for image in images]

            # Forward pass
            predictions = model(images)

            # torchmetrics espera tensores en CPU
            predictions_cpu = [_move_dict_to_cpu(p) for p in predictions]
            targets_cpu = [_move_dict_to_cpu(t) for t in targets]

            # Acumula el batch — aún no calcula
            metric.update(predictions_cpu, targets_cpu)

    # Restaura el modo training si correspondía
    if was_training:
        model.train()

    # Calcula mAP sobre todos los batches y serializa a Python
    return summarize_map_results(metric.compute())



def extract_main_map_metrics(results):
    return {
        "map": results.get("map"),
        "map_50": results.get("map_50"),
        "map_75": results.get("map_75"),
        "mar_100": results.get("mar_100"),
    }
