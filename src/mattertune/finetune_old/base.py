import contextlib
import copy
import fnmatch
import math
import time
from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from functools import cache, partial
from logging import getLogger
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeAlias, cast
import numpy as np
import rich
import rich.console
import rich.markdown
import rich.table
import rich.tree
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from lightning.fabric.utilities.throughput import measure_flops
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRScheduler,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing_extensions import TypeVar, assert_never, override
from pydantic import BaseModel
from mattertune.finetune.scheduler import PerParamGroupLinearWarmupCosineAnnealingRLPLR
from mattertune.finetune.configs import (
    GradientCheckpointingConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    EmbeddingConfig,
    ParamSpecificOptimizerConfig,
    FreezeConfig,
    CheckpointLoadConfig,
    PretrainedCheckpointConfig,
    RLPConfig,
    WarmupCosRLPConfig,
    NormalizationConfig,
    MetricsConfig,
    EMAConfig,
    PositionNoiseAugmentationConfig,
    ResumeCheckpointConfig,
    RLPWarmupConfig
)
from mattertune.finetune.utils import (
    retreive_state_dict_for_finetuning,
    filter_state_dict,
    generate_random_id,
    ScaledSiLU,
)
from mattertune.finetune.metrics import MetricPair, MetricModuleConfig, FinetuneMetricsModule
from mattertune.finetune.criteria import CriteriaConfig
from mattertune.protocol import (
    TBatch,
    BackBoneBaseConfig,
    BackBoneBaseModule,
    BackBoneBaseOutput,
    OutputHeadBaseConfig,
)


log = getLogger(__name__)

class SkipBatch(Exception):
    pass


@contextlib.contextmanager
def _ignore_skip_batch():
    try:
        yield
    except SkipBatch:
        pass

class TestConfig(BaseModel):
    save_checkpoint_base_dir: Path | None = None
    """Where to save the checkpoint information for this run (or None to disable)"""

    save_results_base_dir: Path | None = None
    """Where to save the results for this run (or None to disable)"""


class BatchDumpConfig(BaseModel):
    dump_if_loss_gt: float | None = None
    """Dump the batch if the loss is greater than this value"""

    actsave: bool = False
    """Save the activations using ActSave"""

    dump_cif: bool = False
    """Dump the CIF file"""

    rank_zero_only: bool = False
    """Only dump the batch if rank is zero"""


class FinetuneConfigBase(BaseModel):
    name: str
    """Name of the this series of finetuning tasks"""
    id: str|None = None
    """ID of this finetuning task"""
    base_dir: Path = Path("./FinetuneTasks")
    """Base directory for saving the finetuning task results"""
    backbone: BackBoneBaseConfig 
    """Configuration for the backbone"""
    primary_metric: CriteriaConfig
    """Primary metric to use for early stopping and model selection"""
    edge_dropout: float | None = None
    """Edge dropout probability"""
    dropout: float | None = None
    """Dropout probability"""
    gradient_checkpointing: GradientCheckpointingConfig | None = None
    """Gradient checkpointing configuration"""
    optimizer: OptimizerConfig
    """Optimizer to use."""
    lr_scheduler: LRSchedulerConfig | None = None
    """Learning rate scheduler configuration. If None, no learning rate scheduler is used."""
    embedding: EmbeddingConfig | None = None
    """Configuration for the embedding layer."""
    batch_size: int
    """Batch size to use."""
    eval_batch_size: int | None = None
    """Batch size to use for evaluation. If None, use the same as batch_size."""
    num_workers: int = 8
    """Number of workers to use for data loading."""
    pin_memory: bool = True
    """Whether to use pin memory for data loading."""
    activation_cls: Literal["scaled_silu", "scaled_swish", "silu", "swish"] | None = None
    test: TestConfig | None = None
    """Configuration for test stage"""
    output_heads: list[OutputHeadBaseConfig] = []
    """List of output heads to use"""
    @property
    def targets(self):
        """List of all targets defined in the output heads"""
        return [head.target_name for head in self.output_heads]
    ## TODO: Remove LoRa support for now
    # lora: LoraRootConfig | None = None
    # """Low-rank Adaptation (LoRA) configuration"""
    # dlora: DLoraConfig | None = None
    # """Distributation-Learning of Rank-Adaptation (DLora) configuration"""
    normalization: dict[str, NormalizationConfig] = {}
    """Normalization parameters for each target"""
    parameter_specific_optimizers: list[ParamSpecificOptimizerConfig] | None = None
    """Configuration for parameter-specific optimizers"""
    use_balanced_batch_sampler: bool = False
    """
    Whether to use balanced batch sampler.

    This balances the batches across all distributed nodes (i.e., GPUs, TPUs, nodes, etc.)
    to ensure that each batch has an equal number of **atoms** across all nodes.
    """
    freeze: FreezeConfig = FreezeConfig()
    """Configuration for freezing parameters"""
    ckpt_load: CheckpointLoadConfig = CheckpointLoadConfig()
    """Configuration for behavior when loading checkpoints"""
    shuffle_val: bool = False
    """Whether to shuffle the validation set"""
    shuffle_test: bool = False
    """Whether to shuffle the test set"""
    metrics: MetricModuleConfig = MetricModuleConfig()
    """Configuration for metrics"""
    ema: EMAConfig | None = None
    """Configuration for exponential moving average"""
    debug_print_every: int | None = None
    """Print debug information every `debug_print_every` iterations. `None` to disable."""
    batch_dump: BatchDumpConfig | None = None
    """Configuration for dumping batches"""
    pos_noise_augmentation: PositionNoiseAugmentationConfig | None = None
    """Configuration for adding noise to atomic coordinates"""

    ##TODO: Embedding and BalancedBatchSampler and Dropout
    def __post_init__(self):

        # if self.embedding is None:
        #     self.embedding = EmbeddingConfig(
        #         num_elements=self.backbone.num_elements,
        #         embedding_size=self.backbone.emb_size_atom,
        #     )

        # if self.use_balanced_batch_sampler:
        #     assert not self.trainer.use_distributed_sampler, "config.trainer.use_distributed_sampler must be False when using balanced batch sampler"

        assert self.targets, (
            "At least one target must be specified, "
            f"but none are specified: {self.targets=}"
        )

        # self.backbone.dropout = self.dropout
        # self.backbone.edge_dropout = self.edge_dropout

TConfig = TypeVar("TConfig", bound=FinetuneConfigBase)


class FinetuneModuleBase(pl.LightningModule, Generic[TConfig, TBatch]):
    @override
    def __init__(
        self,
        config: TConfig,
    ):
        """
        Initializes the model with the given configuration.
        """
        super(FinetuneModuleBase, self).__init__()
        if config.id is None:
            config.id = generate_random_id()
        self.config = config
        self.base_dir = config.base_dir / Path(config.name) / Path(config.id)

        # Set up callbacks
        self._callback_constructors = []
        if (ema := self.config.ema) is not None:
            self.register_callback(lambda: ema.construct_callback())

        self._set_rlp_config_monitors()

        self.construct_output_heads()

        ## TODO:
        self.train_metrics = FinetuneMetricsModule(
            self.config.metrics,
            self.metrics_provider,
            self.config.output_heads,
        )
        self.val_metrics = FinetuneMetricsModule(
            self.config.metrics,
            self.metrics_provider,
            self.config.output_heads,
        )
        self.test_metrics = FinetuneMetricsModule(
            self.config.metrics,
            self.metrics_provider,
            self.config.output_heads,
        )

        # Sanity check: ensure all named_parameters have requires_grad=True,
        #   otherwise add them to ignored_parameters.
        self.ignored_parameters = set[nn.Parameter]()
        ignored_parameters_list: list[str] = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                continue
            self.ignored_parameters.add(param)
            ignored_parameters_list.append(name)
        log.info(f"List of ignored parameters: {ignored_parameters_list}")

        self.process_freezing()
        
    @override
    def forward(
        self, 
        data: TBatch
    ):
        backbone_output = self.backbone(data)

        output_head_results = {}

        for output_head_taeget_name, output_head in self.output_heads.items():
            pred = output_head(
                batch_data = data,
                backbone_output = backbone_output,
                output_head_results = output_head_results,
            )
            
        return output_head_results
    
    @property
    def activation_cls(self):
        match self.config.activation_cls:
            case "scaled_silu" | "scaled_swish":
                return ScaledSiLU
            case "silu" | "swish":
                return nn.SiLU
            case None:
                return nn.Identity
            case _:
                raise NotImplementedError(
                    f"Activation {self.backbone.activation=} is not implemented"
                )
    
    def load_backbone(
        self,
        *,
        backbone: BackBoneBaseModule[TBatch, BackBoneBaseOutput],
    ):
        self.backbone = backbone
        
    ## TODO: I think user should provide their own implementation in loading pretrained BackBone
    @classmethod
    def construct_and_load_checkpoint(cls, config: TConfig):
        """
        Constructs the FineTuneModule and loads the checkpoint specified in the config.
        If config.ckpt_load.checkpoint is a PretrainedCheckpointConfig, 
        the model will be constructed and the backbone will be loaded from pretrained checkpoint.
        """
        match config.ckpt_load.checkpoint:
            case PretrainedCheckpointConfig() as ckpt:
                if ckpt.path is None:
                    raise ValueError("No pretrain checkpoint path specified")
                model = cls(config)
                backbone = config.backbone.backbone_cls.load_backbone(str(ckpt.path), **ckpt.checkpoint_load_args)
                model.load_backbone(backbone = backbone)
            case ResumeCheckpointConfig() as ckpt:
                model = cls.load_from_checkpoint(
                    ckpt.path, strict=False, hparams=config
                )
            case None:
                model = cls(config)
            case _:
                assert_never(config.ckpt_load.checkpoint)
        return model
    
    def construct_output_heads(self):
        output_heads = {}
        for head_config in self.config.output_heads:
            if head_config.target_name in output_heads:
                raise ValueError(
                    f"Duplicate target name {head_config.target_name} in output heads"
                )
            output_head = head_config.construct_output_head(
                activation_cls=self.activation_cls,
            )
            output_heads[head_config.target_name] = output_head
        self.output_heads = output_heads
        
    def register_callback(self, callback_constructor):
        """
        Registers a callback constructor to be initialized later.

        Args:
            callback_constructor (Callable): A zero-argument function that returns a callback instance.
        """
        self._callback_constructors.append(callback_constructor)

    def configure_callbacks(self):
        """
        Constructs and returns the list of callbacks for the Trainer.

        Returns:
            List[pl.Callback]: A list of initialized callbacks.
        """
        return [constructor() for constructor in self._callback_constructors]

    @abstractmethod
    def metric_prefix(self) -> str: ...

    def primary_metric(self, split: Literal["train", "val", "test"] | None = "val"):
        if (config := self.config.primary_metric) is None:
            raise ValueError("Primary metric not set in config")
        metric = config.name
        if split is not None:
            metric = f"{split}/{metric}"
        return metric, config.mode

    def _set_rlp_config_monitors(self):
        match self.config.lr_scheduler:
            case RLPConfig(monitor=None) as rlp_config:
                rlp_config.monitor, rlp_config.mode = self.primary_metric()
            case WarmupCosRLPConfig(rlp=RLPConfig(monitor=None) as rlp_config):
                rlp_config.monitor, rlp_config.mode = self.primary_metric()
            case _:
                pass

    def metrics_provider(
        self,
        prop: str,
        batch: TBatch,
        preds: dict[str, torch.Tensor],
    ) -> MetricPair | None:
        if (pred := preds.get(prop)) is None or (
            target := getattr(batch, prop, None)
        ) is None:
            return None

        if (
            self.config.normalization
            and (norm := self.config.normalization.get(prop)) is not None
        ):
            # Denormalize the predictions and targets
            pred = pred * norm.std + norm.mean
            target = target * norm.std + norm.mean

        return MetricPair(predicted=pred, ground_truth=target)
    
    def freeze_backbone(self):
        self.freeze_parameters(self.backbone.parameters(), name="backbone")

    def freeze_parameters(self, parameters: Iterable[nn.Parameter], *, name: str):
        n_params = 0
        for param in parameters:
            if param in self.ignored_parameters:
                continue

            param.requires_grad = False
            n_params += param.numel()
        log.critical(f"Freezing {n_params} parameters in {name}")

    def named_parameters_matching_patterns(
        self,
        patterns: list[str],
        ignored_parameters: set[nn.Parameter] | None = None,
        requires_grad_only: bool = False,
    ):
        ignored_parameters_set = self.ignored_parameters | (ignored_parameters or set())

        for name, param in self.named_parameters():
            if param in ignored_parameters_set:
                continue
            if requires_grad_only and not param.requires_grad:
                continue
            if (
                matching_pattern := next(
                    (pattern for pattern in patterns if fnmatch.fnmatch(name, pattern)),
                    None,
                )
            ) is None:
                continue

            yield name, param, matching_pattern

        all_parameters = [
            param for param in self.parameters() if param not in self.ignored_parameters
        ]
        num_frozen = sum(
            param.numel() for param in all_parameters if not param.requires_grad
        )
        num_train = sum(
            param.numel() for param in all_parameters if param.requires_grad
        )
        num_total = sum(param.numel() for param in all_parameters)
        if num_total:
            percent_frozen = num_frozen / num_total * 100
            log.critical(
                f"Freezing {num_frozen:,} parameters ({percent_frozen:.2f}%) out of "
                f"{num_total:,} total parameters ({num_train:,} trainable)"
            )

    def denormalize_preds(self, preds: dict[str, torch.Tensor]):
        if not self.config.normalization:
            return preds

        return {
            key: (pred * norm.std + norm.mean)
            if (norm := self.config.normalization.get(key)) is not None
            else pred
            for key, pred in preds.items()
        }

    def forward_denormalized(self, data: TBatch, detach: bool = True):
        preds = self(data)
        if detach:
            preds = {
                k: v.detach() if torch.is_tensor(v) else v for k, v in preds.items()
            }
        return self.denormalize_preds(preds)

    def compute_losses(self, batch: TBatch, preds: dict[str, torch.Tensor]):
        losses: list[torch.Tensor] = []

        # Compute losses for graph targets
        for output_head_config in self.config.output_heads:
            if output_head_config.is_classification() and output_head_config.get_num_classes() == 2:
                y_input = preds[output_head_config.target_name]
                if not hasattr(batch, output_head_config.target_name):
                    raise ValueError(
                        f"Batch does not have target {output_head_config.target_name}"
                    )
                y_target = getattr(batch, output_head_config.target_name).float()
                loss = F.binary_cross_entropy_with_logits(y_input, y_target, reduction="sum")
            
            elif output_head_config.is_classification() and output_head_config.get_num_classes() > 2:
                y_input = preds[output_head_config.target_name]
                if not hasattr(batch, output_head_config.target_name):
                    raise ValueError(
                        f"Batch does not have target {output_head_config.target_name}"
                    )
                y_target = getattr(batch, output_head_config.target_name).float()
                weight = None
                if hasattr(output_head_config, "class_weights"):
                    weight = getattr(output_head_config, "class_weights")
                    weight = torch.tensor(weight, device=y_input.device)
                loss = F.cross_entropy(y_input, y_target.long(), weight=weight, reduction="sum")
            
            else:
                y_input = preds[output_head_config.target_name]
                if not hasattr(batch, output_head_config.target_name):
                    raise ValueError(
                        f"Batch does not have target {output_head_config.target_name}"
                    )
                y_target = getattr(batch, output_head_config.target_name).float()
                loss_func_config = output_head_config.loss
                loss = loss_func_config.compute(
                    y_input,
                    y_target,
                )
                
            self.log(f"{output_head_config.target_name}_loss", loss)
            loss = output_head_config.loss_coefficient * loss
            self.log(f"{output_head_config.target_name}_loss_scaled", loss)
            losses.append(loss)

        loss = cast(torch.Tensor, sum(losses))
        self.log("loss", loss)
        return loss

    def _rlp_metric(self, config: RLPConfig):
        monitor = config.monitor
        assert monitor is not None, "RLP monitor must be specified."

        metric_prefix = f"val/{self.metric_prefix()}/"
        assert monitor.startswith(
            metric_prefix
        ), f"RLP {monitor=} must start with {metric_prefix}"
        monitor = monitor[len(metric_prefix) :]

        if (
            monitor.endswith("_mae")
            and (mae_metric := self.val_metrics.maes.get(monitor[: -len("_mae")]))
            is not None
        ):
            return mae_metric

        if (
            monitor.endswith("_balanced_accuracy")
            and (
                cls_metric := self.val_metrics.cls_metrics.get(
                    monitor[: -len("_balanced_accuracy")]
                )
            )
            is not None
        ):
            return cls_metric

        avail_mae_metrics = list(self.val_metrics.maes.keys())
        avail_cls_metrics = list(self.val_metrics.cls_metrics.keys())
        raise ValueError(
            f"RLP monitor {monitor} not found in metrics. "
            f"Available MAE metrics: {avail_mae_metrics}. "
            f"Available classification metrics: {avail_cls_metrics}"
        )

    def _cos_rlp_schedulers(self):
        if (lr_schedulers := self.lr_schedulers()) is None:
            log.warning("No LR scheduler found.")
            return

        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]

        for scheduler in lr_schedulers:
            if isinstance(scheduler, PerParamGroupLinearWarmupCosineAnnealingRLPLR):
                yield scheduler

    def _on_validation_epoch_end_cos_rlp(self, config: WarmupCosRLPConfig):
        rlp_monitor = self._rlp_metric(config.rlp)
        log.debug(f"LR scheduler metrics: {rlp_monitor}")

        metric_value: torch.Tensor | None = None
        for scheduler in self._cos_rlp_schedulers():
            if scheduler.is_in_rlp_stage(self.global_step):
                if metric_value is None:
                    metric_value = rlp_monitor.compute()

                log.debug(f"LR scheduler is in RLP mode. RLP metric: {metric_value}")
                scheduler.rlp_step(metric_value)

    def _on_train_batch_start_cos_rlp(self):
        for scheduler in self._cos_rlp_schedulers():
            scheduler.on_new_step(self.global_step)

    @override
    def on_train_batch_start(self, batch: TBatch, batch_idx: int):
        match self.config.lr_scheduler:
            case WarmupCosRLPConfig():
                self._on_train_batch_start_cos_rlp()
            case _:
                pass

    @override
    def on_validation_epoch_end(self):
        match self.config.lr_scheduler:
            case WarmupCosRLPConfig() as config:
                self._on_validation_epoch_end_cos_rlp(config)
            case _:
                pass

    @override
    def training_step(self, batch: TBatch, batch_idx: int):
        try:
            with self.log_context(prefix=f"train/{self.metric_prefix()}/"):
                preds = self(batch)

                loss = self.compute_losses(batch, preds)
                self._process_batch_dump(batch, batch_idx, loss)

                self.log_dict(self.train_metrics(batch, preds))
                return loss
        except SkipBatch:
            log.warning(f"Batch #{batch_idx} {batch} skipped.")
            return self.zero_loss()

    @override
    def validation_step(self, batch: TBatch, batch_idx: int):
        with _ignore_skip_batch(), self.log_context(
            prefix=f"val/{self.metric_prefix()}/"
        ):
            preds = self(batch)

            self.log_dict(self.val_metrics(batch, preds))

    @override
    def test_step(self, batch: TBatch, batch_idx: int):
        with _ignore_skip_batch(), self.log_context(
            prefix=f"test/{self.metric_prefix()}/"
        ):
            preds = self(batch)

            self.log_dict(self.test_metrics(batch, preds))

    @override
    def predict_step(self, batch: TBatch, batch_idx: int):
        preds = self(batch)
        return preds

    def outhead_parameters(self):
        head_params = (
            list(self.graph_outputs.parameters())
            + list(self.node_outputs.parameters())
            + list(self.graph_classification_outputs.parameters())
        )
        return head_params

    def backbone_outhead_parameters(
        self,
    ):
        main_params = list(self.parameters())
        head_params = self.outhead_parameters()
        head_params_set = set(head_params)
        main_params = [p for p in main_params if p not in head_params_set]
        return main_params, head_params

    def _warmup_step(
        self,
        config: RLPWarmupConfig,
        optimizer: torch.optim.Optimizer | LightningOptimizer,
    ):
        # Compute the current step
        match config.step_type:
            case "step":
                current_step = self.global_step
            case "epoch":
                current_step = self.current_epoch
            case _:
                assert_never(config.step_type)

        if current_step > config.steps:
            return

        initial_lr = self.config.optimizer.lr
        lr_scale = min(1.0, float(current_step + 1) / config.steps)
        for pg in optimizer.param_groups:
            pg["lr"] = initial_lr * lr_scale

    @override
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        match self.config.lr_scheduler:
            case RLPConfig(warmup=RLPWarmupConfig() as warmup):
                self._warmup_step(warmup, optimizer)
            case _:
                pass

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def split_parameters(
        self,
        pattern_lists: list[list[str]],
        no_double_counting: bool = True,
        requires_grad_only: bool = True,
        on_unused: Literal["error", "warn", "ignore"] = "warn",
    ):
        """
        Splits the parameters of the model into multiple groups based on the provided pattern lists.

        Args:
            pattern_lists (list[list[str]]): A list of pattern lists. Each pattern list contains a set of patterns
                used to match parameter names.
            no_double_counting (bool): If True, parameters that match multiple patterns will only be counted once.
            requires_grad_only (bool): If True, only parameters with requires_grad=True will be considered.
            on_unused (Literal["error","warn", "ignore"]): What to do if no parameters match the pattern. Options are:

        Returns:
            parameters (list[list[nn.Parameter]]): A list of parameter groups. Each group contains the parameters
                that match the patterns in the corresponding pattern list.
            all_parameters (list[nn.Parameter]): The remaining parameters that do not match any of the patterns.
        """

        matched_parameters = set[nn.Parameter]()
        all_parameters = [
            p for p in self.parameters() if not requires_grad_only or p.requires_grad
        ]

        parameters: list[list[nn.Parameter]] = []
        for patterns in pattern_lists:
            matching = [
                p
                for _, p, _ in self.named_parameters_matching_patterns(
                    patterns,
                    ignored_parameters=matched_parameters,
                    requires_grad_only=requires_grad_only,
                )
            ]
            log.info(f"Matched parameters for patterns {patterns}: {len(matching)}")

            parameters.append(matching)

            # Remove matching parameters from all_parameters.
            all_parameters = [
                p for p in all_parameters if all(p is not m for m in matching)
            ]

            # If no_double_counting is True, add the matching parameters to the set of matched parameters.
            if no_double_counting:
                matched_parameters.update(matching)

            # If no parameters matched the pattern, raise an error or warning.
            if not matching:
                error_msg = f"No parameters matched the pattern list: {patterns}"
                match on_unused:
                    case "error":
                        raise ValueError(error_msg)
                    case "warn":
                        log.warning(error_msg)
                    case "ignore":
                        pass
                    case _:
                        assert_never(on_unused)

        return parameters, all_parameters

    def _cos_annealing_hparams(
        self, lr_config: WarmupCosRLPConfig, *, lr_initial: float
    ):
        if (warmup_steps := lr_config.warmup_steps) is None:
            if warmup_epochs := lr_config.warmup_epochs:
                assert warmup_epochs >= 0, f"Invalid warmup_epochs: {warmup_epochs}"
                _ = self.trainer.estimated_stepping_batches  # make sure dataloaders are loaded for self.trainer.num_training_batches
                num_steps_per_epoch = math.ceil(
                    self.trainer.num_training_batches
                    / self.trainer.accumulate_grad_batches
                )
                warmup_steps = int(warmup_epochs * num_steps_per_epoch)
            else:
                warmup_steps = 0
        log.critical(f"Computed warmup_steps: {warmup_steps}")

        if not (max_steps := lr_config.max_steps):
            if max_epochs := lr_config.max_epochs:
                _ = self.trainer.estimated_stepping_batches  # make sure dataloaders are loaded for self.trainer.num_training_batches
                num_steps_per_epoch = math.ceil(
                    self.trainer.num_training_batches
                    / self.trainer.accumulate_grad_batches
                )
                max_steps = int(max_epochs * num_steps_per_epoch)
            else:
                max_steps = self.trainer.estimated_stepping_batches
                assert math.isfinite(max_steps), f"{max_steps=} is not finite"
                max_steps = int(max_steps)

        log.critical(f"Computed max_steps: {max_steps}")

        assert (
            lr_config.min_lr_factor > 0 and lr_config.min_lr_factor <= 1
        ), f"Invalid {lr_config.min_lr_factor=}"
        min_lr = lr_initial * lr_config.min_lr_factor

        assert (
            lr_config.warmup_start_lr_factor > 0
            and lr_config.warmup_start_lr_factor <= 1
        ), f"Invalid {lr_config.warmup_start_lr_factor=}"
        warmup_start_lr = lr_initial * lr_config.warmup_start_lr_factor

        lr_scheduler_hparams = dict(
            warmup_epochs=warmup_steps,
            max_epochs=max_steps,
            warmup_start_lr=warmup_start_lr,
            eta_min=min_lr,
            should_restart=lr_config.should_restart,
        )

        return lr_scheduler_hparams

    def _construct_lr_scheduler(
        self, optimizer: torch.optim.Optimizer, config: RLPConfig
    ) -> LRSchedulerConfigType:
        assert config.monitor is not None, f"{config=}"
        assert config.mode is not None, f"{config=}"

        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            patience=config.patience,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps,
            verbose=True,
        )

        return {
            "scheduler": lr_scheduler,
            "monitor": config.monitor,
            "interval": config.interval,
            "frequency": config.frequency,
            "strict": True,  # type: ignore
        }

    def configure_optimizers_param_specific_optimizers(
        self, configs: list[ParamSpecificOptimizerConfig]
    ):
        params_list, rest_params = self.split_parameters(
            [c.paremeter_patterns for c in configs],
            on_unused="error",
        )
        optimizer = optimizer_from_config(
            [
                *(
                    (
                        self.config.optimizer if c.optimizer is None else c.optimizer,
                        params,
                        c.name or ",".join(c.paremeter_patterns),
                    )
                    for c, params in zip(configs, params_list)
                    # Ignore empty parameter groups
                    if params
                ),
                (self.config.optimizer, rest_params, "rest"),
            ],
            base=self.config.optimizer,
        )

        out: OptimizerLRScheduler = {
            "optimizer": optimizer,
        }
        if (lr_config := self.config.lr_scheduler) is None:
            return out

        match lr_config:
            case RLPConfig():
                assert all(
                    c.lr_scheduler is None for c in configs
                ), f"lr_scheduler is not None for some configs: {configs=}"

                if (
                    lr_scheduler := self._construct_lr_scheduler(optimizer, lr_config)
                ) is not None:
                    out["lr_scheduler"] = lr_scheduler
            case WarmupCosRLPConfig():
                param_group_lr_scheduler_settings = [
                    *(
                        self._cos_annealing_hparams(
                            (
                                lr_config
                                if c.lr_scheduler is None
                                or not isinstance(c.lr_scheduler, WarmupCosRLPConfig)
                                else c.lr_scheduler
                            ),
                            lr_initial=param_group["lr"],
                        )
                        for c, param_group in zip(configs, optimizer.param_groups[:-1])
                    ),
                    self._cos_annealing_hparams(
                        lr_config, lr_initial=optimizer.param_groups[-1]["lr"]
                    ),
                ]

                log.critical(f"{param_group_lr_scheduler_settings=}")
                lr_scheduler = PerParamGroupLinearWarmupCosineAnnealingRLPLR(
                    optimizer,
                    param_group_lr_scheduler_settings,
                    lr_config.rlp._to_linear_warmup_cos_rlp_dict(),
                    max_epochs=next(
                        (s["max_epochs"] for s in param_group_lr_scheduler_settings)
                    ),
                )
                out["lr_scheduler"] = {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            case _:
                assert_never(lr_config)

        return out

    def _report_parameters(self):
        trainable_parameters: list[str] = []
        non_trainable_parameters: list[str] = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_parameters.append(name)
            else:
                non_trainable_parameters.append(name)

        trainable_parameters_str = "\n".join(f"\t-{p}" for p in trainable_parameters)
        log.debug(f"Trainable parameters {trainable_parameters_str}")

        non_trainable_parameters_str = "\n".join(
            f"\t-{p}" for p in non_trainable_parameters
        )
        log.debug(f"Non-trainable parameters {non_trainable_parameters_str}")

    @override
    def configure_optimizers(self):
        if self.config.parameter_specific_optimizers is not None:
            out = self.configure_optimizers_param_specific_optimizers(
                self.config.parameter_specific_optimizers
            )
            self._report_parameters()
            return out

        optimizer = optimizer_from_config(
            [(self.config.optimizer, self.parameters())],
        )

        out: OptimizerLRScheduler = {
            "optimizer": optimizer,
        }
        if (lr_config := self.config.lr_scheduler) is None:
            return out

        assert isinstance(
            lr_config, RLPConfig
        ), "Only RLPConfig is supported if `parameter_specific_optimizers` is None"
        if (
            lr_scheduler := self._construct_lr_scheduler(optimizer, lr_config)
        ) is not None:
            out["lr_scheduler"] = lr_scheduler

        return out
