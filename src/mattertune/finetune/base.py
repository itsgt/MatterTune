import torch
import torch.nn as nn
import torchmetrics
from ase import Atoms
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Generic
from mattertune.data_structures import TMatterTuneData, TMatterTuneBatch, RawDataProviderBaseConfig, MatterTuneDataSetBase
from mattertune.finetune.backbone import BackBoneBaseConfig
from mattertune.output_heads.base import OutputHeadBaseConfig
from mattertune.finetune.metrics import MetricsModuleConfig, MetricsModule
from mattertune.finetune.optimizer import OptimizerConfig
from mattertune.finetune.lr_scheduler import LRSchedulerConfig
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, OptimizerLRSchedulerConfig
from pytorch_lightning.trainer.states import TrainerFn
from contextlib import ExitStack
from typing_extensions import TypeVar, override, cast, Sequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, DistributedSampler


class FinetuneModuleBaseConfig(BaseModel):
    project: str = "finetune"
    """Name of this series of finetuning experiments."""
    run_name: str = "default_run"
    """Name of this specific run."""
    backbone: BackBoneBaseConfig
    """Backbone model configuration."""
    output_heads: Sequence[OutputHeadBaseConfig]
    """Output heads configurations."""
    raw_data_provider: RawDataProviderBaseConfig|None = None
    """Raw data provider configuration."""
    metrics_module: MetricsModuleConfig
    """Metrics module configuration."""
    optimizer: OptimizerConfig
    """Optimizer."""
    lr_scheduler: LRSchedulerConfig
    """Learning Rate Scheduler"""
    batch_size: int
    """Batch size."""
    num_workers: int = 0
    """Number of workers for data loading."""
    pin_memory: bool = True
    """Whether to pin memory for data loading."""
    ignore_data_errors: bool = True
    """Whether to ignore data processing errors during training."""
    early_stopping_patience: int|None
    """Number of epochs to wait before early stopping. Set to None to disable early stopping."""
    early_stopping_min_delta: float = 0.0
    """Minimum change in the primary metric to consider as an improvement."""
    early_stopping_mode: str = "min"
    """Mode for early stopping. One of ["min", "max"]."""
    
    
TFinetuneModuleConfig = TypeVar("TFinetuneModuleConfig", bound=FinetuneModuleBaseConfig)


class FinetuneModuleBase(pl.LightningModule, Generic[TMatterTuneBatch, TFinetuneModuleConfig]):
    """
    Finetune module base class. Heritate from pytorch_lightning.LightningModule.
    """
    def __init__(
        self,
        config: TFinetuneModuleConfig,
    ):
        super().__init__()
        self.config = config
        self.raw_data_provider = config.raw_data_provider
        config.raw_data_provider = None ## We don't want to save raw_data_provider in the checkpoint
        self.backbone = config.backbone.construct_backbone()
        
        assert len(config.output_heads) > 0, "At least one output head is required."
        ## Put Energy Head First, and Put Stress Head before Force Head
        energy_output_heads = []
        force_output_heads = []
        stress_output_heads = []
        other_output_heads = []
        self.compel_grad_enabled = False
        for output_head_config in config.output_heads:
            if "energy" in output_head_config.target_name:
                energy_output_heads.append(output_head_config)
            elif "stress" in output_head_config.target_name:
                stress_output_heads.append(output_head_config)
            elif "force" in output_head_config.target_name:
                force_output_heads.append(output_head_config)
            else:
                other_output_heads.append(output_head_config)
            compel_grad_enabled_i = output_head_config.compel_grad_enabled
            self.compel_grad_enabled = self.compel_grad_enabled or compel_grad_enabled_i
        config.output_heads = energy_output_heads + stress_output_heads + force_output_heads + other_output_heads
        self.output_heads = nn.ModuleDict()
        for output_head_config in config.output_heads:
            output_head = output_head_config.construct_output_head()
            self.output_heads[output_head_config.target_name] = output_head
        
        self.ignore_data_errors = config.ignore_data_errors
        self.metrics_module = config.metrics_module.construct_metrics_module()
        self._freeze_backbone_and_output_heads()
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No parameters require gradients. Please ensure that some parts of the model are trainable.")
        self.enable_callbacks = True
        self.save_hyperparameters(ignore=["raw_data_provider"])
        
    def disable_callbacks(self):
        """
        Disable all callbacks.
        """
        self.enable_callbacks = False
        
    def configure_callbacks(self) -> list[pl.Callback]:
        """
        Configure callbacks for the trainer.
        """
        callbacks = []
        if self.enable_callbacks:
            primary_metric_name = self.metrics_module.get_primary_metric_name()
            
            # Setup checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.trainer.default_root_dir,  # use default root dir
                filename=f"best-{self.config.run_name}-{{epoch:02d}}-{{val_loss:.2f}}",
                save_top_k=1,
                monitor=f"val/{primary_metric_name}",  # monitor primary metric
                mode=self.config.early_stopping_mode,
                save_last=True,  # always save the last checkpoint
            )
            callbacks.append(checkpoint_callback)
            
            # Setup early stopping callback if enabled
            if self.config.early_stopping_patience is not None:
                early_stopping_callback = EarlyStopping(
                    monitor=f"val/{primary_metric_name}",
                    min_delta=self.config.early_stopping_min_delta,
                    patience=self.config.early_stopping_patience,
                    mode=self.config.early_stopping_mode,
                    verbose=False,
                )
                callbacks.append(early_stopping_callback)
        return callbacks
        
            
    def setup(self, stage: str):
        """
        Setup the model for training, validation, and testing.
        Here backbone's process_raw() method is called to process the raw data.
        """
        if stage in (TrainerFn.FITTING, TrainerFn.VALIDATING):
            if self.raw_data_provider is None:
                raise ValueError("Raw data provider is required for training and validation.")
            data_provider = self.raw_data_provider.build_provider()
            train_raw_data, train_labels = data_provider.get_train_data()
            val_raw_data, val_labels = data_provider.get_val_data()
            train_data_list = [self.backbone.process_raw(atoms=atoms, idx = i, labels=labels, inference=False) for i, (atoms, labels) in enumerate(zip(train_raw_data, train_labels))]
            val_data_list = [self.backbone.process_raw(atoms=atoms, idx = i, labels=labels, inference=False) for i, (atoms, labels) in enumerate(zip(val_raw_data, val_labels))]
            self.train_dataset = MatterTuneDataSetBase(train_data_list)
            self.val_dataset = MatterTuneDataSetBase(val_data_list)
        elif stage == TrainerFn.TESTING:
            if self.raw_data_provider is None:
                raise ValueError("Raw data provider is required for testing.")
            data_provider = self.raw_data_provider.build_provider()
            test_raw_data, test_labels = data_provider.get_test_data()
            test_data_list = [self.backbone.process_raw(atoms=atoms, idx = i, labels=labels, inference=False) for i, (atoms, labels) in enumerate(zip(test_raw_data, test_labels))]
            self.test_dataset = MatterTuneDataSetBase(test_data_list)
        elif stage == TrainerFn.PREDICTING:
            ## We are not going to build predict dataset here
            pass
        else:
            raise ValueError(f"Invalid stage: {stage}")
        
    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset) if self.trainer and self.trainer.num_devices > 1 else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None) and True,
            num_workers=self.config.num_workers,
            collate_fn=self.backbone.collate_fn,
            sampler=sampler,
            pin_memory=self.config.pin_memory,
        )
    
    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset) if self.trainer and self.trainer.num_devices > 1 else None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None) and False,
            num_workers=self.config.num_workers,
            collate_fn=self.backbone.collate_fn,
            sampler=sampler,
            pin_memory=self.config.pin_memory,
        )
        
    def test_dataloader(self):
        sampler = DistributedSampler(self.test_dataset) if self.trainer and self.trainer.num_devices > 1 else None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None) and False,
            num_workers=self.config.num_workers,
            collate_fn=self.backbone.collate_fn,
            sampler=sampler,
            pin_memory=self.config.pin_memory,
        )
    
    def forward(
        self,
        batch: TMatterTuneBatch,
    ) -> dict[str, torch.Tensor]:
        
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
        batch: TMatterTuneBatch,
        batch_idx: int,
    ):
        output_head_results = self(batch)
        
        ## Compute loss and Log into monitor
        loss_sum = None
        loss_results: dict[str, float] = {}
        batch_size = int(torch.max(batch.batch).detach().cpu().item() + 1)
        batch_labels = batch.labels
        for output_head_config in self.config.output_heads:
            target_name = output_head_config.target_name
            pred = output_head_results[target_name]
            target = batch_labels[target_name]
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
        batch: TMatterTuneBatch,
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
            
    @override
    def on_validation_epoch_end(self):
        return super().on_validation_epoch_end()
        
    @override
    def test_step(
        self,
        batch: TMatterTuneBatch,
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
    def predict_step(
        self,
        batch: TMatterTuneBatch,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        output_head_results = self(batch)
        output_head_results["idx"] = batch.idx
        return output_head_results
    
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
        
    @property
    def inference_mode(self) -> bool:
        # Return False when compel_grad_enabled is True
        return not self.compel_grad_enabled