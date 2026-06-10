from __future__ import annotations

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def _resolve_fasterrcnn_weights(weights):
    if weights is None:
        return None

    if weights == "DEFAULT":
        return FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    if isinstance(weights, str):
        return FasterRCNN_ResNet50_FPN_Weights[weights]

    return weights


def replace_fasterrcnn_predictor(model, num_classes: int):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def create_fasterrcnn_model(
    num_classes: int,
    weights="DEFAULT",
    trainable_backbone_layers: int = 3,
    min_size=None,
    max_size=None,
):
    resolved_weights = _resolve_fasterrcnn_weights(weights)

    model_kwargs = {
        "weights": resolved_weights,
        "trainable_backbone_layers": trainable_backbone_layers,
    }

    if min_size is not None:
        model_kwargs["min_size"] = min_size
    if max_size is not None:
        model_kwargs["max_size"] = max_size

    model = fasterrcnn_resnet50_fpn(**model_kwargs)
    return replace_fasterrcnn_predictor(model, num_classes=num_classes)


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def describe_parameter_counts(model):
    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    frozen = total - trainable
    return {
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "total_parameters": total,
    }


def build_fasterrcnn_variants(num_classes: int):
    return {
        "fasterrcnn_head_only": {
            "model_name": "fasterrcnn",
            "weights": "DEFAULT",
            "trainable_backbone_layers": 0,
            "num_classes": num_classes,
        },
        "fasterrcnn_partial_backbone": {
            "model_name": "fasterrcnn",
            "weights": "DEFAULT",
            "trainable_backbone_layers": 2,
            "num_classes": num_classes,
        },
        "fasterrcnn_full_backbone": {
            "model_name": "fasterrcnn",
            "weights": "DEFAULT",
            "trainable_backbone_layers": 5,
            "num_classes": num_classes,
        },
    }


def create_model_from_config(config):
    model_name = config.get("model_name", "fasterrcnn")

    if model_name != "fasterrcnn":
        raise ValueError(f"Modelo no soportado todavia: {model_name}")

    return create_fasterrcnn_model(
        num_classes=config["num_classes"],
        weights=config.get("weights", "DEFAULT"),
        trainable_backbone_layers=config.get("trainable_backbone_layers", 3),
        min_size=config.get("min_size"),
        max_size=config.get("max_size"),
    )


def get_trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_optimizer(model, optimizer_name="sgd", lr=0.005, weight_decay=0.0005, momentum=0.9):
    parameters = get_trainable_parameters(model)

    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Optimizador no soportado: {optimizer_name}")


def build_scheduler(optimizer, scheduler_name=None, step_size=3, gamma=0.1, patience=2, factor=0.1):
    if scheduler_name is None:
        return None

    normalized = scheduler_name.lower()

    if normalized == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if normalized == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=factor,
            patience=patience,
        )

    raise ValueError(f"Scheduler no soportado: {scheduler_name}")
