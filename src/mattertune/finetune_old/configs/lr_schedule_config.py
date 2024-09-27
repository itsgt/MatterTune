from typing import Annotated, Literal, TypeAlias
from pydantic import BaseModel, Field


class RLPWarmupConfig(BaseModel):
    step_type: Literal["step", "epoch"]
    """The type of step to use for the warmup"""

    steps: int
    """Number of steps for the warmup"""

    start_lr_factor: float
    """The factor to multiply the initial learning rate by at the start of the warmup"""


class RLPConfig(BaseModel):
    name: Literal["rlp"] = "rlp"

    monitor: str | None = None
    mode: Literal["min", "max"] | None = None
    patience: int = 10
    factor: float = 0.1
    min_lr: float = 0.0
    eps: float = 1.0e-8
    cooldown: int = 0
    threshold: float = 1.0e-4
    threshold_mode: Literal["rel", "abs"] = "rel"
    interval: Literal["epoch", "step"] = "epoch"
    frequency: int = 1
    warmup: RLPWarmupConfig | None = None

    def _to_linear_warmup_cos_rlp_dict(self):
        """
        Params for PerParamGroupLinearWarmupCosineAnnealingRLPLR's RLP
            mode="min",
            factor=0.1,
            patience=10,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-8,
            verbose=False,
        """
        return {
            "mode": self.mode,
            "factor": self.factor,
            "patience": self.patience,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "cooldown": self.cooldown,
            "min_lr": self.min_lr,
            "eps": self.eps,
            "verbose": True,
        }


class WarmupCosRLPConfig(BaseModel):
    name: Literal["warmup_cos_rlp"] = "warmup_cos_rlp"

    warmup_steps: int | None = None
    warmup_epochs: int | float | None = None
    max_steps: int | None = None
    max_epochs: int | float | None = None
    warmup_start_lr_factor: float = 0.0
    min_lr_factor: float = 1.0e-2
    last_step: int = -1
    should_restart: bool = False
    
    rlp: RLPConfig
    
    def __post_init__(self):
        assert self.rlp.warmup is None, "WarmupCosRLPConfig cannot have a warmup in the RLP config"
        

LRSchedulerConfig: TypeAlias = Annotated[
    RLPConfig | WarmupCosRLPConfig, Field(discriminator="name")
]