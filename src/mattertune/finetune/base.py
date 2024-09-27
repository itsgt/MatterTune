from typing import Generic
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from pydantic import BaseModel
from typing_extensions import TypeVar, override, cast
import torch
import torch.nn as nn
from mattertune.protocol import (
    TBatch,
    BackBoneBaseConfig,
    BackBoneBaseModule,
    BackBoneBaseOutput,
    OutputHeadBaseConfig,
)
from mattertune.finetune.utils import (
    generate_random_id,
)
from mattertune.finetune.monitor import MonitorConfig
from mattertune.finetune.optimizer import OptimizerBaseConfig
from mattertune.finetune.lr_scheduler import LRSchedulerBaseConfig


class FinetuneModuleBaseConfig(BaseModel):
    name: str = "finetune"
    """Name of this series of finetuning experiments."""
    id: str|None = None
    """ID of this finetuning task."""
    backbone: BackBoneBaseConfig
    """Backbone model configuration."""
    output_heads: list[OutputHeadBaseConfig]
    """Output heads configurations."""
    monitor: MonitorConfig
    """Monitor for early stopping and logging."""
    optimizer: OptimizerBaseConfig
    """Optimizer."""
    lr_scheduler: LRSchedulerBaseConfig
    """Learning rate scheduler."""
    
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
        if self.config.id is None:
            self.config.id = generate_random_id()
        self.backbone = config.backbone.construct_backbone()
        assert len(config.output_heads) > 0, "At least one output head is required."
        self.output_heads: dict[str, nn.Module] = {}
        for output_head_config in config.output_heads:
            output_head = output_head_config.construct_output_head()
            self.output_heads[output_head_config.target_name] = output_head
        self.monitor = config.monitor.construct_monitor()
    
    def forward(
        self,
        batch: TBatch,
    ):
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
        for output_head_config in self.config.output_heads:
            target_name = output_head_config.target_name
            pred = output_head_results[target_name]
            if not hasattr(batch, target_name):
                raise ValueError(f"Target {target_name} not found in batch.")
            target = getattr(batch, target_name)
            loss = output_head_config.loss.compute_impl(pred, target)
            if hasattr(output_head_config, "loss_coefficient"):
                loss_coefficient = output_head_config.loss_coefficient
            else:
                loss_coefficient = 1.0
                
            loss_results[target_name+"-"+output_head_config.loss.name] = loss.item() * loss_coefficient
                
            if loss_sum is None:
                loss_sum = loss * loss_coefficient
            else:
                loss_sum += loss * loss_coefficient
        assert loss_sum is not None, "Found loss=None, At least one loss is required."
        self.log("train/total-loss", loss_sum.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.monitor.log_losses_step_end(
            batch=batch,
            loss_results=loss_results,
            step_type="train",
        )
        
        ## Evaluate metrics and Log into monitor
        metrics_results = self.monitor.evaluate(
            batch=batch,
            output_head_results=output_head_results,
        )
        self.monitor.log_metrics_step_end(
            batch=batch,
            metrics_results=metrics_results,
            step_type="train",
        )
        
        return loss_sum
    
    @override
    def on_train_epoch_end(self):
        current_epoch = self.current_epoch
        self.monitor.log_metrics_epoch_end(
            epoch_type = "train",
            current_epoch=current_epoch,
        )
        self.monitor.log_losses_epoch_end(
            epoch_type = "train",
            current_epoch=current_epoch,
        )
        return super().on_train_epoch_end()
    
    
    @override
    def validation_step(
        self,
        batch: TBatch,
        batch_idx: int,
    ):
        output_head_results = self(batch)
        metrics_results = self.monitor.evaluate(
            batch=batch,
            output_head_results=output_head_results,
        )
        self.monitor.log_metrics_step_end(
            batch=batch,
            metrics_results=metrics_results,
            step_type="val",
        )
            
    @override
    def on_validation_epoch_end(self):
        ## Update Logger with Both Metrics and Losses in Validation
        current_epoch = self.current_epoch
        self.monitor.log_metrics_epoch_end(
            epoch_type = "val",
            current_epoch=current_epoch,
        )
        self.monitor.log_losses_epoch_end(
            epoch_type = "val",
            current_epoch=current_epoch,
        )
        ## Update Early Stopping if Necessary
        if self.monitor.early_stopping and self.monitor.perform_early_stop:
            self.trainer.should_stop = True
        return super().on_validation_epoch_end()
        
            
    @override
    def test_step(
        self,
        batch: TBatch,
        batch_idx: int,
    ):
        output_head_results = self(batch)
        metrics_results = self.monitor.evaluate(
            batch=batch,
            output_head_results=output_head_results,
        )
        self.monitor.log_metrics_step_end(
            batch=batch,
            metrics_results=metrics_results,
            step_type="test",
        )
    
    @override
    def on_test_epoch_end(self):
        current_epoch = self.current_epoch
        self.monitor.log_metrics_epoch_end(
            epoch_type = "test",
            current_epoch=current_epoch,
        )
        self.monitor.log_losses_epoch_end(
            epoch_type = "test",
            current_epoch=current_epoch,
        )
        return super().on_test_epoch_end()
    
    @override
    def configure_optimizers(self):
        parameters = []
        if not self.config.backbone.freeze:
            parameters += list(self.backbone.parameters())
        else:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        for output_head_config in self.config.output_heads:
            if not output_head_config.freeze:
                target_name = output_head_config.target_name
                parameters += list(self.output_heads[target_name].parameters())
            else:
                self.output_heads[output_head_config.target_name].eval()
                for param in self.output_heads[output_head_config.target_name].parameters():
                    param.requires_grad = False
        optimizer = self.config.optimizer.construct_optimizer(
            parameters=parameters,   
        )
        lr_scheduler = self.config.lr_scheduler.construct_lr_scheduler(
            optimizer=optimizer,
        )
        return cast(OptimizerLRSchedulerConfig, {"optimizer": optimizer, "lr_scheduler": lr_scheduler})