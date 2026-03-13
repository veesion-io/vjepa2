from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def jepa_forward(
    data: torch.Tensor,
    encoder: nn.Module,
    classifiers: list[nn.Module],
    is_training: bool,
) -> tuple[list, torch.Tensor | list[torch.Tensor]]:
    with torch.no_grad():
        encoder_outputs = encoder([[data]])
        if isinstance(encoder_outputs, list):
            encoder_outputs = (
                encoder_outputs[0] if len(encoder_outputs) == 1 else encoder_outputs
            )

    if not is_training:
        with torch.no_grad():
            if isinstance(encoder_outputs, list):
                outputs = [[c(o) for o in encoder_outputs] for c in classifiers]
            else:
                outputs = [c(encoder_outputs) for c in classifiers]
    else:
        if isinstance(encoder_outputs, list):
            outputs = [[c(o) for o in encoder_outputs] for c in classifiers]
        else:
            outputs = [c(encoder_outputs) for c in classifiers]

    return outputs, encoder_outputs


def compute_losses(
    outputs: list,
    labels: torch.Tensor,
    encoder_outputs: torch.Tensor | list[torch.Tensor],
    criterion: nn.Module,
) -> list[torch.Tensor] | list[list[torch.Tensor]]:
    if isinstance(encoder_outputs, list):
        return [[criterion(o, labels) for o in c_outputs] for c_outputs in outputs]
    return [criterion(o, labels) for o in outputs]


def compute_accuracies(
    outputs: list,
    labels: torch.Tensor,
    encoder_outputs: torch.Tensor | list[torch.Tensor],
    batch_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    with torch.no_grad():
        if isinstance(encoder_outputs, list):
            aggregated_outputs = [
                sum([F.softmax(o, dim=1) for o in c_outputs]) / len(c_outputs)
                for c_outputs in outputs
            ]
        else:
            aggregated_outputs = [F.softmax(o, dim=1) for o in outputs]

        top1_accs = [
            100.0 * out.max(dim=1).indices.eq(labels).sum() / batch_size
            for out in aggregated_outputs
        ]

    return aggregated_outputs, top1_accs


def jepa_backward(
    losses: list[torch.Tensor] | list[list[torch.Tensor]],
    optimizers: list[torch.optim.Optimizer],
    scalers: list[torch.cuda.amp.GradScaler | None],
    encoder_outputs: torch.Tensor | list[torch.Tensor],
) -> None:
    if isinstance(encoder_outputs, list):
        for scaler, classifier_losses, optimizer in zip(scalers, losses, optimizers):
            optimizer.zero_grad()
            if scaler is not None:
                for loss in classifier_losses:
                    scaler.scale(loss).backward(retain_graph=True)
                scaler.step(optimizer)
                scaler.update()
            else:
                for loss in classifier_losses:
                    loss.backward(retain_graph=True)
                optimizer.step()
    else:
        for scaler, loss, optimizer in zip(scalers, losses, optimizers):
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()


def select_best_classifier(
    classifier_metrics: list[dict],
    outputs: list,
    losses: list,
    encoder_outputs: torch.Tensor | list[torch.Tensor],
    aggregated_outputs: list[torch.Tensor],
    subset: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | float]:
    best_idx = np.argmax([m[subset].avg for m in classifier_metrics])
    best_output = aggregated_outputs[best_idx]
    _, best_predictions = torch.max(best_output.data, 1)

    if isinstance(encoder_outputs, list):
        best_raw_output = outputs[best_idx][0]
        best_loss = sum(losses[best_idx]) / len(losses[best_idx])
    else:
        best_raw_output = outputs[best_idx]
        best_loss = losses[best_idx]

    return best_predictions, best_raw_output, best_loss
