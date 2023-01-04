from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from src.lib.metrics import pF1Beta


class LightningBaseModule(pl.LightningModule):
    def __init__(
        self, num_classes: int, backbone: nn.Module, supervised_criterion: nn.Module
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.supervised_criterion = supervised_criterion
        self.metrics: dict[str, Any]
        self.init_metrics()

    def init_metrics(self) -> None:
        self.metrics = {
            "accuracy": torchmetrics.Accuracy(
                task="multiclass", num_classes=self.num_classes
            ),
            "f1-score": torchmetrics.F1Score(
                task="multiclass", num_classes=self.num_classes
            ),
        }
        if self.num_classes > 5:
            self.metrics["top5-accuracy"] = torchmetrics.Accuracy(
                task="multiclass", num_classes=self.num_classes, top_k=5
            )

        for metric_name, metric in self.metrics.items():
            self.add_module(f"metrics_{metric_name}", metric)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.backbone(x)
        return out

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        return self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        return self._shared_eval(batch, batch_idx, "test")

    def validation_epoch_end(
        self,
        outputs: list[torch.Tensor | dict[str, Any]]
        | list[list[torch.Tensor | dict[str, Any]]],
    ) -> None:
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(
        self,
        outputs: list[torch.Tensor | dict[str, Any]]
        | list[list[torch.Tensor | dict[str, Any]]],
    ) -> None:
        self._shared_epoch_end(outputs, "test")

    def training_epoch_end(
        self,
        outputs: list[torch.Tensor | dict[str, Any]]
        | list[list[torch.Tensor | dict[str, Any]]],
    ) -> None:
        self._shared_epoch_end(outputs, "train")

    def _shared_eval(
        self, batch: torch.Tensor, batch_idx: int, prefix: str
    ) -> dict[str, torch.Tensor]:
        x, y = batch
        logits = self.backbone(x)
        preds = logits.softmax(dim=1)
        loss = self.supervised_criterion(preds, y)
        for metric_name, metric in self.metrics.items():
            metric.update(preds, y)
        return {
            "loss": loss,
        }

    def _shared_epoch_end(
        self,
        outs: list[torch.Tensor | dict[str, Any]]
        | list[list[torch.Tensor | dict[str, Any]]],
        prefix: str,
    ) -> None:
        losses = []
        for o in outs:
            losses.append(o["loss"] if isinstance(o, dict) else o)
        loss = torch.stack(losses).mean()
        self.log(f"{prefix}/loss", loss)
        for metric_name, metric in self.metrics.items():
            metric_value = metric.compute()
            self.log(f"{prefix}/{metric_name}", metric_value)
            if metric_name == "accuracy":
                self.log(f"{prefix}/error", 1 - metric_value)


class LightningClassifier(LightningBaseModule):
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        supervised_criterion: nn.Module = pF1Beta(),
        lr: float = 1e-1,
    ):
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            supervised_criterion=supervised_criterion,
        )

        self.lr = lr
        self.save_hyperparameters("lr")  # type: ignore

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.SGD(self.backbone.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/accuracy",
                "interval": "epoch",
                "frequency": self.trainer.check_val_every_n_epoch,
            },
        }

    def training_step(
        self, train_batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        inputs, labels = train_batch
        logits = self.backbone(inputs)
        preds = logits.softmax(dim=1)

        # Loss
        loss = self.supervised_criterion(preds, labels)
        self.log("train_batch/loss", loss)
        return {
            "loss": loss,
        }
