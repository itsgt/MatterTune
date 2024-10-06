import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError, CosineSimilarity, Accuracy, Precision, Recall, F1Score
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Generic
from mattertune.protocol import TBatch, BackBoneBaseConfig
from mattertune.output_heads.base import OutputHeadBaseConfig
from mattertune.finetune.utils import generate_random_id
from mattertune.finetune.metrics import MetricsModuleConfig, MetricsModule
from mattertune.finetune.optimizer import OptimizerConfig
from mattertune.finetune.lr_scheduler import LRSchedulerConfig
from mattertune.finetune.data_module import MatterTuneDataModuleBase
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from contextlib import ExitStack
from typing_extensions import TypeVar, override, cast, Sequence
import pytorch_lightning as pl


class EarlyStoppingModule:
    """
    A module to handle early stopping based on a primary metric.
    """
    def __init__(
        self,
        patience: int,
        min_delta: float,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric = None
        self.wait = 0
        self.perform_early_stop = False

    def update(self, current_metric: float) -> None:
        if self.best_metric is None:
            self.best_metric = current_metric
        else:
            if (self.mode == "min" and current_metric < self.best_metric - self.min_delta) or \
               (self.mode == "max" and current_metric > self.best_metric + self.min_delta):
                self.best_metric = current_metric
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.perform_early_stop = True


class FinetuneModuleBaseConfig(BaseModel):
    project: str = "finetune"
    """Name of this series of finetuning experiments."""
    run_name: str = "default_run"
    """Name of this specific run."""
    backbone: BackBoneBaseConfig
    """Backbone model configuration."""
    output_heads: Sequence[OutputHeadBaseConfig]
    """Output heads configurations."""
    metrics_module: MetricsModuleConfig
    """Metrics module configuration."""
    optimizer: OptimizerConfig
    """Optimizer."""
    lr_scheduler: LRSchedulerConfig
    """Learning Rate Scheduler"""
    ignore_data_errors: bool = True
    """Whether to ignore data processing errors during training."""
    early_stopping_patience: int|None
    """Number of epochs to wait before early stopping. Set to None to disable early stopping."""
    early_stopping_min_delta: float = 0.0
    """Minimum change in the primary metric to consider as an improvement."""
    early_stopping_mode: str = "min"
    """Mode for early stopping. One of ["min", "max"]."""
    
    
TFinetuneModuleConfig = TypeVar("TFinetuneModuleConfig", bound=FinetuneModuleBaseConfig)


class FinetuneModuleBase(pl.LightningModule, Generic[TBatch, TFinetuneModuleConfig]):
    """
    Finetune module base class. Heritate from pytorch_lightning.LightningModule.
    """
    def __init__(
        self,
        config: TFinetuneModuleConfig,
    ):
        super().__init__()
        self.config = config
        self.backbone = config.backbone.construct_backbone()
        assert len(config.output_heads) > 0, "At least one output head is required."
        
        ## Put Energy Head First, and Put Stress Head before Force Head
        energy_output_heads = []
        force_output_heads = []
        stress_output_heads = []
        other_output_heads = []
        for output_head_config in config.output_heads:
            if "energy" in output_head_config.target_name:
                energy_output_heads.append(output_head_config)
            elif "stress" in output_head_config.target_name:
                stress_output_heads.append(output_head_config)
            elif "force" in output_head_config.target_name:
                force_output_heads.append(output_head_config)
            else:
                other_output_heads.append(output_head_config)
        config.output_heads = energy_output_heads + stress_output_heads + force_output_heads + other_output_heads
        self.output_heads = nn.ModuleDict()
        for output_head_config in config.output_heads:
            output_head = output_head_config.construct_output_head()
            self.output_heads[output_head_config.target_name] = output_head
        
        self.ignore_data_errors = config.ignore_data_errors
        self.metrics_module = config.metrics_module.construct_metrics_module()
        self._freeze_backbone_and_output_heads()
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No parameters require gradients. Please ensure that some parts of the model are trainable.")
        
        # Initialize EarlyStoppingModule if enabled
        if config.early_stopping_patience is not None:
            self.early_stopping = EarlyStoppingModule(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                mode=config.early_stopping_mode,
            )
            self.primary_metric_mean = torchmetrics.MeanMetric()
        else:
            self.early_stopping = None
            self.primary_metric_mean = None
    
    def forward(
        self,
        batch: TBatch,
    ):
        
        with ExitStack() as stack:
            # Enter all the necessary contexts for output heads.
            # Right now, this is only for gradient forces, which
            #   requires torch.inference_mode(False), torch.enable_grad,
            #   and data.pos.requires_grad_(True).
            for output_head in self.config.output_heads:
                stack.enter_context(output_head.model_forward_context(data=batch))
                
            if self.ignore_data_errors:
                try:
                    batch = self.backbone.process_batch_under_grad(batch, training=True)
                except Exception as e:
                    print(f"Error in forward pass: {e}")
            else:
                batch = self.backbone.process_batch_under_grad(batch, training=True)

            backbone_output = self.backbone(batch)
            output_head_results = {}

            for target_name, output_head in self.output_heads.items():
                output = output_head(
                    batch_data=batch,
                    backbone_output=backbone_output,
                    output_head_results=output_head_results,
                )
                if target_name not in output_head_results:
                    output_head_results[target_name] = output
            return output_head_results

    @override
    def training_step(
        self,
        batch: TBatch,
        batch_idx: int,
    ):
        output_head_results = self(batch)
        
        ## Compute loss and Log into monitor
        loss_sum = None
        loss_results: dict[str, float] = {}
        batch_size = int(torch.max(batch.batch).detach().cpu().item() + 1)
        for output_head_config in self.config.output_heads:
            target_name = output_head_config.target_name
            pred = output_head_results[target_name]
            if not hasattr(batch, target_name):
                raise ValueError(f"Target {target_name} not found in batch.")
            target = getattr(batch, target_name)
            loss = output_head_config.loss.compute(pred, target, batch)
            if hasattr(output_head_config, "loss_coefficient"):
                loss_coefficient = output_head_config.loss_coefficient
            else:
                loss_coefficient = 1.0
                
            loss_results[target_name+"-"+output_head_config.loss.name] = loss.detach().cpu().item() * loss_coefficient
                
            if loss_sum is None:
                loss_sum = loss * loss_coefficient
            else:
                loss_sum += loss * loss_coefficient
        assert loss_sum is not None, "Found loss=None, At least one loss is required."
        self.log("train/total-loss", loss_sum.detach().cpu().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        for loss_name, loss_value in loss_results.items():
            self.log(f"train/{loss_name}", loss_value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
            
        ## Evaluate metrics and Log into monitor
        metrics_results = self.metrics_module.compute(
            batch=batch,
            output_head_results=output_head_results,
        )
        batch_size = int(torch.max(batch.batch).detach().cpu().item() + 1)
        for metric_name, metric_value in metrics_results.items():
            self.log(f"train/{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        
        return loss_sum
    
    @override
    def on_train_epoch_end(self):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return super().on_train_epoch_end()
    
    @override
    def validation_step(
        self,
        batch: TBatch,
        batch_idx: int,
    ):
        output_head_results = self(batch)
        metrics_results = self.metrics_module.compute(
            batch=batch,
            output_head_results=output_head_results,
        )
        batch_size = int(torch.max(batch.batch).detach().cpu().item() + 1)
        for metric_name, metric_value in metrics_results.items():
            self.log(f"val/{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
            
        # Track primary metric for early stopping if enabled
        if self.early_stopping is not None and self.primary_metric_mean is not None:
            primary_metric_name = self.config.metrics_module.primary_metric.target_name+"-"+self.config.metrics_module.primary_metric.metric_calculator.name
            if self.config.metrics_module.primary_metric.normalize_by_num_atoms:
                primary_metric_name += "-peratom"
            primary_metric_value = metrics_results[primary_metric_name]
            self.primary_metric_mean.update(primary_metric_value)
            self.log("val/primary_metric", primary_metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
    @override
    def on_validation_epoch_end(self):
        # Early stopping check
        if self.early_stopping is not None and self.primary_metric_mean is not None:
            primary_metric_avg = self.primary_metric_mean.compute()
            self.early_stopping.update(primary_metric_avg.detach().cpu().item())
            self.primary_metric_mean.reset()
            if self.early_stopping.perform_early_stop:
                self.trainer.should_stop = True
        return super().on_validation_epoch_end()
        
    @override
    def test_step(
        self,
        batch: TBatch,
        batch_idx: int,
    ):
        output_head_results = self(batch)
        metrics_results = self.metrics_module.compute(
            batch=batch,
            output_head_results=output_head_results,
        )
        batch_size = int(torch.max(batch.batch).detach().cpu().item() + 1)
        for metric_name, metric_value in metrics_results.items():
            self.log(f"test/{metric_name}", metric_value, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
    
    @override
    def on_test_epoch_end(self):
        return super().on_test_epoch_end()
    
    @override
    def configure_optimizers(self):
        parameters = list(self.backbone.parameters())
        for output_head_config in self.config.output_heads:
            target_name = output_head_config.target_name
            parameters += list(self.output_heads[target_name].parameters())
        optimizer = self.config.optimizer.construct_optimizer(
            parameters=parameters,   
        )
        
        lr_scheduler = self.config.lr_scheduler.construct_lr_scheduler(
            optimizer=optimizer,
        )
        return cast(OptimizerLRSchedulerConfig, {"optimizer": optimizer, "lr_scheduler": lr_scheduler})
    
    def _freeze_backbone_and_output_heads(self):
        if self.config.backbone.freeze:
            print("Freezing Backbone Model.")
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            print("Unfreezing Backbone Model.")
            self.backbone.train()
            for param in self.backbone.parameters():
                param.requires_grad = True
        for output_head_config in self.config.output_heads:
            if output_head_config.freeze:
                print(f"Freezing Output Head {output_head_config.target_name}.")
                self.output_heads[output_head_config.target_name].eval()
                for param in self.output_heads[output_head_config.target_name].parameters():
                    param.requires_grad = False
            else:
                print(f"Unfreezing Output Head {output_head_config.target_name}.")
                self.output_heads[output_head_config.target_name].train()
                for param in self.output_heads[output_head_config.target_name].parameters():
                    param.requires_grad = True