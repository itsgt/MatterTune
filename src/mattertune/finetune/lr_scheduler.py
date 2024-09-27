from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Any
from pydantic import BaseModel
import torch


class LRSchedulerBase(torch.optim.lr_scheduler._LRScheduler, ABC):
    @override
    def step(self, epoch: int|None = None, metrics: Any = None): ...

class LRSchedulerBaseConfig(BaseModel, ABC):
    @abstractmethod
    def construct_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> LRSchedulerBase: ...

class StepLRConfig(LRSchedulerBaseConfig):
    name: str = "StepLR"
    """Name of the learning rate scheduler."""
    step_size: int
    """Period of learning rate decay."""
    gamma: float
    """Multiplicative factor of learning rate decay."""
    
    @override
    def construct_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ):
        return StepLRScheduler(
            optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
        )
        
class StepLRScheduler(LRSchedulerBase):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float,
    ):
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    
    @override
    def step(self, epoch: int|None = None, metrics: Any = None):
        self.scheduler.step(epoch)


class MultiStepLRConfig(LRSchedulerBaseConfig):
    name: str = "MultiStepLR"
    """Name of the learning rate scheduler."""
    milestones: list[int]
    """List of epoch indices. Must be increasing."""
    gamma: float
    """Multiplicative factor of learning rate decay."""
    
    def construct_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ):
        return MultiStepLRScheduler(
            optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
        )
        
class MultiStepLRScheduler(LRSchedulerBase):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: list[int],
        gamma: float,
    ):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    
    def step(self, epoch: int|None = None, metrics: Any = None):
        self.scheduler.step(epoch)
    

class ExponentialConfig(LRSchedulerBaseConfig):
    name: str = "Exponential"
    """Name of the learning rate scheduler."""
    gamma: float
    """Multiplicative factor of learning rate decay."""
    
    def construct_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ):
        return ExponentialLRScheduler(
            optimizer,
            gamma=self.gamma,
        )
        

class ExponentialLRScheduler(LRSchedulerBase):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: float,
    ):
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
        )
    
    def step(self, epoch: int|None = None, metrics: Any = None):
        self.scheduler.step(epoch)


class ReduceOnPlateauConfig(LRSchedulerBaseConfig):
    name: str = "ReduceOnPlateau"
    """Name of the learning rate scheduler."""
    mode: str
    """One of {"min", "max"}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing."""
    factor: float
    """Factor by which the learning rate will be reduced. new_lr = lr * factor."""
    patience: int
    """Number of epochs with no improvement after which learning rate will be reduced."""
    threshold: float = 1e-4
    """Threshold for measuring the new optimum, to only focus on significant changes."""
    threshold_mode: str = "rel"
    """One of {"rel", "abs"}. In rel mode, dynamic_threshold = best * (1 + threshold) in ‘max’ mode or best * (1 - threshold) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode."""
    cooldown: int = 0
    """Number of epochs to wait before resuming normal operation after lr has been reduced."""
    min_lr: float = 0
    """A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively."""
    eps: float = 1e-8
    """Threshold for testing the new optimum, to only focus on significant changes."""
    
    def construct_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ):
        return ReduceOnPlateauLRScheduler(
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

class ReduceOnPlateauLRScheduler(LRSchedulerBase):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str,
        factor: float,
        patience: int,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
    ):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )
    
    def step(self, epoch: int|None = None, metrics: Any = None):
        self.scheduler.step(metrics)
        
class CosineAnnealingLRConfig(LRSchedulerBaseConfig):
    name: str = "CosineAnnealingLR"
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
    ):
        return CosineAnnealingLRScheduler(
            optimizer,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch,
        )

class CosineAnnealingLRScheduler(LRSchedulerBase):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
    
    def step(self, epoch: int|None = None, metrics: Any = None):
        self.scheduler.step(epoch)
