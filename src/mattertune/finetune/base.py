from __future__ import annotations

import contextlib
import logging
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from typing import Any, Generic, Protocol, runtime_checkable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from typing_extensions import TypeVar, cast, override

from mattertune.finetune.lr_scheduler import LRSchedulerConfig
from mattertune.finetune.optimizer import OptimizerConfig
from mattertune.protocol import TBatch, TData

from .loss import (
    HuberLossConfig,
    L2MAELossConfig,
    MAELossConfig,
    MSELossConfig,
    l2_mae_loss,
)
from .metrics import FinetuneMetrics
from .properties import PropertyConfig

log = logging.getLogger(__name__)


@runtime_checkable
class ForwardContextPropertyHeadProtocol(Protocol[TBatch]):
    def model_forward_context(
        self, data: TBatch
    ) -> contextlib.AbstractContextManager: ...


class FinetuneModuleBaseConfig(BaseModel):
    project: str = "finetune"
    """Name of this series of finetuning experiments."""
    run_name: str = "default_run"
    """Name of this specific run."""

    optimizer: OptimizerConfig
    """Optimizer."""
    lr_scheduler: LRSchedulerConfig
    """Learning Rate Scheduler"""

    ignore_data_errors: bool = True
    """Whether to ignore data processing errors during training."""

    properties: Mapping[str, PropertyConfig]
    """Properties to predict."""


TFinetuneModuleConfig = TypeVar("TFinetuneModuleConfig", bound=FinetuneModuleBaseConfig)


class FinetuneModuleBase(
    pl.LightningModule, Generic[TData, TBatch, TFinetuneModuleConfig]
):
    """
    Finetune module base class. Heritate from pytorch_lightning.LightningModule.
    """

    # region ABC methods for data processing
    @abstractmethod
    def cpu_data_transform(self, data: TData) -> TData:
        """
        Transform data (on the CPU) before being batched and sent to the GPU.
        """
        ...

    @abstractmethod
    def collate_fn(self, data_list: list[TData]) -> TBatch:
        """
        Collate function for the DataLoader
        """
        ...

    @abstractmethod
    def gpu_batch_transform(self, batch: TBatch) -> TBatch:
        """
        Transform batch (on the GPU) before being fed to the model.
        """
        ...

    @abstractmethod
    def ground_truth_dict_from_batch(self, batch: TBatch) -> dict[str, torch.Tensor]:
        """
        Extract ground truth values from a batch. The output of this function
        should be a dictionary with keys corresponding to the target names
        and values corresponding to the ground truth values. The values should
        be torch tensors that match, in shape, the output of the corresponding
        output head.
        """
        ...

    # endregion

    # region ABC methods for output heads and model forward pass
    @abstractmethod
    def create_head(self, property_config: PropertyConfig) -> nn.Module:
        """
        Create an output head for a given property.
        """
        ...

    @abstractmethod
    def model_forward_context(self, data: TBatch) -> contextlib.AbstractContextManager:
        """
        Context manager for the model forward pass.
        """
        ...

    @abstractmethod
    def forward_backbone(self, batch: TBatch) -> Any:
        """
        Forward pass of the backbone model.
        """
        ...

    # endregion

    # region Overridable methods with sensible defaults
    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        """
        Return the parameters of the backbone model.

        Default implementation returns all parameters that are not part of any output head.
        """
        head_params = set(p for head in self.heads.values() for p in head.parameters())
        for p in self.parameters():
            if p not in head_params:
                yield p

    # endregion

    def __init__(self, config: TFinetuneModuleConfig):
        super().__init__()

        self.config = config
        self.heads = nn.ModuleDict()

        def _sort_properties(property_config_tuple: tuple[str, PropertyConfig]):
            _, property_config = property_config_tuple
            if property_config.type == "energy":
                return 0
            if property_config.type == "stress":
                return 1
            if property_config.type == "force":
                return 2
            return 3

        for property_name, property_config in sorted(
            list(self.config.properties.items()), key=_sort_properties
        ):
            head = self.create_head(property_config)
            self.heads[property_name] = head

        # Create metrics
        self.train_metrics = FinetuneMetrics(self.config.properties)
        self.val_metrics = FinetuneMetrics(self.config.properties)
        self.test_metrics = FinetuneMetrics(self.config.properties)

        if not any(p for p in self.parameters() if p.requires_grad):
            raise ValueError(
                "No parameters require gradients. Please ensure that some parts of the model are trainable."
            )

    def forward(self, batch: TBatch) -> dict[str, torch.Tensor]:
        with ExitStack() as stack:
            # Enter all the necessary contexts for output heads.
            # Right now, this is only for gradient forces, which
            #   requires torch.inference_mode(False), torch.enable_grad,
            #   and data.pos.requires_grad_(True).
            for output_head in self.heads.values():
                if isinstance(output_head, ForwardContextPropertyHeadProtocol):
                    stack.enter_context(output_head.model_forward_context(batch))

            # Generate graph/etc
            if self.config.ignore_data_errors:
                try:
                    batch = self.gpu_batch_transform(batch)
                except Exception as e:
                    log.warning("Error in forward pass.", exc_info=e)
            else:
                batch = self.gpu_batch_transform(batch)

            # Run the model through the backbone
            backbone_output = self.forward_backbone(batch)

            # Run the model through the output heads
            head_results: dict[str, torch.Tensor] = {}
            for property_name, head in self.heads.items():
                head_results[property_name] = head(batch, backbone_output)

            return head_results

    def _compute_loss_for_head(
        self,
        config: PropertyConfig,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ):
        match config.loss:
            case MAELossConfig():
                return F.l1_loss(prediction, target)
            case MSELossConfig():
                return F.mse_loss(prediction, target)
            case HuberLossConfig():
                return F.huber_loss(prediction, target, delta=config.loss.delta)
            case L2MAELossConfig():
                return l2_mae_loss(prediction, target)
            case _:
                raise ValueError(f"Unknown loss function: {config.loss}")

    def _compute_loss(
        self,
        prediction: dict[str, torch.Tensor],
        ground_truth: dict[str, torch.Tensor],
        log: bool = True,
        log_prefix: str = "",
    ):
        losses: list[torch.Tensor] = []
        for target_name, head_config in self.config.properties.items():
            # Get the target and prediction
            pred = prediction[target_name]
            target = ground_truth[target_name]

            # Compute the loss
            loss = (
                self._compute_loss_for_head(head_config, pred, target)
                * head_config.loss_coefficient
            )

            # Log the loss
            if log:
                self.log(f"{log_prefix}{target_name}_loss", loss)
            losses.append(loss)

        # Sum the losses
        loss = cast(torch.Tensor, sum(losses))

        # Log the total loss & return
        if log:
            self.log(f"{log_prefix}total_loss", loss)
        return loss

    def _common_step(
        self,
        batch: TBatch,
        name: str,
        metrics: FinetuneMetrics | None,
        log: bool = True,
    ):
        prediction = self(batch)
        ground_truth = self.ground_truth_dict_from_batch(batch)

        # Compute loss
        loss = self._compute_loss(
            prediction,
            ground_truth,
            log=log,
            log_prefix=f"{name}/",
        )

        # Log metrics
        if log and metrics is not None:
            self.log_dict(
                {
                    f"{name}/{metric_name}": metric
                    for metric_name, metric in metrics(prediction, ground_truth).items()
                }
            )

        return prediction, loss

    @override
    def training_step(self, batch: TBatch, batch_idx: int):
        _, loss = self._common_step(batch, "train", self.train_metrics)
        return loss

    @override
    def validation_step(self, batch: TBatch, batch_idx: int):
        _ = self._common_step(batch, "val", self.val_metrics)

    @override
    def test_step(self, batch: TBatch, batch_idx: int):
        _ = self._common_step(batch, "test", self.test_metrics)

    @override
    def predict_step(self, batch: TBatch, batch_idx: int):
        prediction, _ = self._common_step(batch, "predict", None, log=False)
        return prediction

    @override
    def configure_optimizers(self):
        optimizer = self.config.optimizer.construct_optimizer(self.parameters())
        lr_scheduler = self.config.lr_scheduler.construct_lr_scheduler(optimizer)
        return cast(
            OptimizerLRSchedulerConfig,
            {"optimizer": optimizer, "lr_scheduler": lr_scheduler},
        )
