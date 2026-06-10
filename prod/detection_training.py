from __future__ import annotations

import copy
from pathlib import Path

import torch

from .detection_metrics import evaluate_map, extract_main_map_metrics


def move_targets_to_device(targets, device):
    return [
        {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def move_batch_to_device(images, targets, device):
    images = [image.to(device) for image in images]
    targets = move_targets_to_device(targets, device)
    return images, targets


def compute_loss_dict(model, images, targets):
    return model(images, targets)


def train_one_epoch(model, dataloader, optimizer, device, epoch_index, max_batches=None):
    model.train()
    running_loss = 0.0
    running_steps = 0

    for batch_index, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images, targets = move_batch_to_device(images, targets, device)

        loss_dict = compute_loss_dict(model, images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += float(total_loss.item())
        running_steps += 1

    average_loss = running_loss / max(running_steps, 1)
    return {
        "epoch": epoch_index,
        "train_loss": average_loss,
        "train_steps": running_steps,
    }


def evaluate_detection_loss(model, dataloader, device, max_batches=None):
    was_training = model.training
    model.train()
    running_loss = 0.0
    running_steps = 0

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images, targets = move_batch_to_device(images, targets, device)
            loss_dict = compute_loss_dict(model, images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            running_loss += float(total_loss.item())
            running_steps += 1

    if not was_training:
        model.eval()

    average_loss = running_loss / max(running_steps, 1)
    return {
        "val_loss": average_loss,
        "val_steps": running_steps,
    }


def _scheduler_step(scheduler, metrics):
    if scheduler is None:
        return

    if scheduler.__class__.__name__ == "ReduceLROnPlateau":
        scheduler.step(metrics.get("map", 0.0))
        return

    scheduler.step()


def _checkpoint_payload(model, optimizer, scheduler, history, epoch, experiment_name, config):
    payload = {
        "experiment_name": experiment_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": config,
    }

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    return payload


def run_detection_experiment(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    experiment_name,
    config,
    scheduler=None,
    output_dir=None,
    max_train_batches=None,
    max_val_batches=None,
    class_metrics=True,
):
    history = []
    best_metric = float("-inf")
    best_epoch = None
    best_checkpoint_path = None
    best_payload = None

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_index=epoch,
            max_batches=max_train_batches,
        )
        val_loss_metrics = evaluate_detection_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            max_batches=max_val_batches,
        )
        map_metrics = evaluate_map(
            model=model,
            dataloader=val_loader,
            device=device,
            class_metrics=class_metrics,
            max_batches=max_val_batches,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "val_loss": val_loss_metrics["val_loss"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            **extract_main_map_metrics(map_metrics),
        }
        history.append(epoch_metrics)

        current_metric = epoch_metrics.get("map")
        if current_metric is not None and current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            best_payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=copy.deepcopy(history),
                epoch=epoch,
                experiment_name=experiment_name,
                config=config,
            )

            if output_path is not None:
                best_checkpoint_path = output_path / f"{experiment_name}_best.pth"
                torch.save(best_payload, best_checkpoint_path)

        _scheduler_step(scheduler, epoch_metrics)

    return {
        "history": history,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
        "best_payload": best_payload,
    }


def load_checkpoint(model, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
