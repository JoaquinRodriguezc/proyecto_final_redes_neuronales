from __future__ import annotations

import torch
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    fcos_resnet50_fpn,
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def _resolve_weights(weights, weight_enum):
    if weights is None:
        return None
    if weights == "DEFAULT":
        return weight_enum.DEFAULT
    if isinstance(weights, str):
        return weight_enum[weights]
    return weights


def replace_fasterrcnn_predictor(model, num_classes: int):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def replace_retinanet_head(model, num_classes: int):
    classification_head = model.head.classification_head
    in_channels = classification_head.cls_logits.in_channels
    num_anchors = classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    return model


def replace_fcos_head(model, num_classes: int):
    classification_head = model.head.classification_head
    in_channels = classification_head.cls_logits.in_channels
    num_anchors = classification_head.num_anchors
    num_convs = len(classification_head.conv)
    model.head.classification_head = FCOSClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        num_convs=num_convs,
    )
    return model


_MODEL_REGISTRY = {
    "fasterrcnn": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, replace_fasterrcnn_predictor),
    "retinanet":  (retinanet_resnet50_fpn,  RetinaNet_ResNet50_FPN_Weights,  replace_retinanet_head),
    "fcos":       (fcos_resnet50_fpn,        FCOS_ResNet50_FPN_Weights,       replace_fcos_head),
}


def _create_model(
    model_name: str,
    num_classes: int,
    weights="DEFAULT",
    trainable_backbone_layers: int = 3,
    weights_backbone=None,
    min_size=None,
    max_size=None,
):
    factory, weight_enum, replace_head = _MODEL_REGISTRY[model_name]
    resolved_weights = _resolve_weights(weights, weight_enum)

    model_kwargs = {
        "weights": resolved_weights,
        "trainable_backbone_layers": trainable_backbone_layers,
        "weights_backbone": weights_backbone,
    }
    if min_size is not None:
        model_kwargs["min_size"] = min_size
    if max_size is not None:
        model_kwargs["max_size"] = max_size

    return replace_head(factory(**model_kwargs), num_classes=num_classes)


def create_model_from_config(config: dict):
    model_name = config.get("model_name", "fasterrcnn")
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Modelo no soportado todavia: {model_name}")
    return _create_model(
        model_name=model_name,
        num_classes=config["num_classes"],
        weights=config.get("weights", "DEFAULT"),
        trainable_backbone_layers=config.get("trainable_backbone_layers", 3),
        weights_backbone=config.get("weights_backbone"),
        min_size=config.get("min_size"),
        max_size=config.get("max_size"),
    )


def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def describe_parameter_counts(model) -> dict:
    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    return {
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "total_parameters": total,
    }


def build_fasterrcnn_variants(num_classes: int) -> dict:
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


def get_trainable_parameters(model) -> list:
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(model, optimizer_name="sgd", lr=0.005, weight_decay=0.0005, momentum=0.9):
    parameters = get_trainable_parameters(model)

    if optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Optimizador no soportado: {optimizer_name}")
