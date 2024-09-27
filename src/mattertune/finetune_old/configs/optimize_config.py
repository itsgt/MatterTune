from typing import Any, Literal, TypeAlias, Annotated
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import torch.nn as nn
import torch.optim as optim
from mattertune.finetune.configs import LRSchedulerConfig


class OutputConfig(BaseModel):
    num_mlps: int = 5
    """Number of MLPs in the output layer."""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""


class AdamWConfig(BaseModel):
    name: Literal["adamw"] = "adamw"

    lr: float
    """Learning rate for the optimizer."""

    weight_decay: float = 1.0e-2
    """Weight decay (L2 penalty) for the optimizer."""

    betas: tuple[float, float] = (0.9, 0.999)
    """
    Betas for the optimizer:
    (beta1, beta2) are the coefficients used for computing running averages of
    gradient and its square.
    """

    eps: float = 1e-8
    """Term added to the denominator to improve numerical stability."""

    amsgrad: bool = False
    """Whether to use the AMSGrad variant of this algorithm."""


@dataclass(frozen=True)
class _OptimizerParamGroupConfig:
    cls: type[optim.Optimizer]
    param_group_kwargs: dict[str, Any] = field(default_factory=lambda: {})
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {})


OptimizerConfig: TypeAlias = Annotated[AdamWConfig, Field(discriminator="name")]


class FreezeConfig(BaseModel):
    backbone: bool = False
    """Should the backbone be frozen?"""
    embedding: bool = False
    """Should the embedding layer be frozen?"""

    backbone_bases: bool = False
    """Should the basis functions in the backbone be frozen?"""
    backbone_interaction_layers: list[int] | None = None
    """Which interaction layers, if any, in the backbone should be frozen?"""
    backbone_output_layers: list[int] | None = None
    """Which output layers, if any, in the backbone should be frozen?"""

    parameter_patterns: list[str] = []
    """List of parameter patterns to freeze"""

    ensure_non_frozen_parameter_patterns: list[str] = []
    """List of parameter patterns to ensure are not frozen"""

    report_parameters: bool = False
    """
    If `True`, we will print a large table of all parameters and their requires_grad status.
    """


class ParamSpecificOptimizerConfig(BaseModel):
    name: str | None = None
    """The name of the parameter group for this config"""

    paremeter_patterns: list[str] = []
    """List of parameter patterns to match for this config"""

    optimizer: OptimizerConfig | None = None
    """
    The optimizer config for this parameter group.
    If None, the default optimizer will be used.
    """

    lr_scheduler: LRSchedulerConfig | None = None
    """
    The learning rate scheduler config for this parameter group.
    If None, the default learning rate scheduler will be used.
    """