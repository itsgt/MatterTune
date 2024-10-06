from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Literal, TypeAlias, Annotated
from pydantic import BaseModel, Field
import torch


class LRSchedulerBaseConfig(BaseModel, ABC):
    @abstractmethod
    def construct_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Abstract method to construct a learning rate scheduler."""
        pass


class StepLRConfig(LRSchedulerBaseConfig):
    name: Literal["StepLR"] = "StepLR"
    """Name of the learning rate scheduler."""
    step_size: int
    """Period of learning rate decay."""
    gamma: float
    """Multiplicative factor of learning rate decay."""

    @override
    def construct_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.StepLR:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
        )


class MultiStepLRConfig(LRSchedulerBaseConfig):
    name: Literal["MultiStepLR"] = "MultiStepLR"
    """Name of the learning rate scheduler."""
    milestones: list[int]
    """List of epoch indices. Must be increasing."""
    gamma: float
    """Multiplicative factor of learning rate decay."""

    def construct_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.MultiStepLR:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
        )


class ExponentialConfig(LRSchedulerBaseConfig):
    name: Literal["ExponentialLR"] = "ExponentialLR"
    """Name of the learning rate scheduler."""
    gamma: float
    """Multiplicative factor of learning rate decay."""

    def construct_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.ExponentialLR:
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.gamma,
        )


class ReduceOnPlateauConfig(LRSchedulerBaseConfig):
    name: Literal["ReduceLROnPlateau"] = "ReduceLROnPlateau"
    """Name of the learning rate scheduler."""
    mode: str
    """One of {"min", "max"}. Determines when to reduce the learning rate."""
    factor: float
    """Factor by which the learning rate will be reduced."""
    patience: int
    """Number of epochs with no improvement after which learning rate will be reduced."""
    threshold: float = 1e-4
    """Threshold for measuring the new optimum."""
    threshold_mode: str = "rel"
    """One of {"rel", "abs"}. Determines the threshold mode."""
    cooldown: int = 0
    """Number of epochs to wait before resuming normal operation."""
    min_lr: float = 0
    """A lower bound on the learning rate."""
    eps: float = 1e-8
    """Threshold for testing the new optimum."""

    def construct_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
            threshold_mode=self.threshold_mode,
            cooldown=self.cooldown,
            min_lr=self.min_lr,
            eps=self.eps,
        )


class CosineAnnealingLRConfig(LRSchedulerBaseConfig):
    name: Literal["CosineAnnealingLR"] = "CosineAnnealingLR"
    """Name of the learning rate scheduler."""
    T_max: int
    """Maximum number of iterations."""
    eta_min: float = 0
    """Minimum learning rate."""
    last_epoch: int = -1
    """The index of last epoch."""

    def construct_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch,
        )


LRSchedulerConfig: TypeAlias = Annotated[
    StepLRConfig
    | MultiStepLRConfig
    | ExponentialConfig
    | ReduceOnPlateauConfig
    | CosineAnnealingLRConfig,
    Field(discriminator="name"),
]