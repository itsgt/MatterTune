from abc import ABC, abstractmethod
from typing import Generic, Literal
from pydantic import BaseModel
from mattertune.protocol import TBatch
import torch
import torch.nn as nn
import torchmetrics
from mattertune.finetune.logger import _BaseLogger

class _BaseMetrics(nn.Module, ABC):
    method: str
    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor: ...
    
class MAEMetric(_BaseMetrics):
    method = "MAE"
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torchmetrics.functional.mean_absolute_error(pred, target)
    
class RMSEMetric(_BaseMetrics):
    method = "RMSE"
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torchmetrics.functional.mean_squared_error(pred, target).sqrt()

class CosineSimilarityMetric(_BaseMetrics):
    method = "CosineSimilarity"
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torchmetrics.functional.cosine_similarity(pred, target)

class AccuracyMetric(_BaseMetrics):
    method = "Accuracy"
    def __init__(
        self,
       num_classes: int,
    ):
        super().__init__()
        assert num_classes > 1, "Number of classes must be greater than 1."
        self.num_classes = num_classes
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_classes == 2:
            return torchmetrics.functional.accuracy(pred, target, task="binary", num_classes=self.num_classes)
        else:
            return torchmetrics.functional.accuracy(pred, target, task="multiclass", num_classes=self.num_classes)

class PrecisionMetric(_BaseMetrics):
    method = "Precision"
    def __init__(
        self,
       num_classes: int,
    ):
        super().__init__()
        assert num_classes > 1, "Number of classes must be greater than 1."
        self.num_classes = num_classes
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_classes == 2:
            return torchmetrics.functional.precision(pred, target, task="binary", num_classes=self.num_classes)
        else:
            return torchmetrics.functional.precision(pred, target, task="multiclass", num_classes=self.num_classes)

class RecallMetric(_BaseMetrics):
    method = "Recall"
    def __init__(
        self,
       num_classes: int,
    ):
        super().__init__()
        assert num_classes > 1, "Number of classes must be greater than 1."
        self.num_classes = num_classes
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_classes == 2:
            return torchmetrics.functional.recall(pred, target, task="binary", num_classes=self.num_classes)
        else:
            return torchmetrics.functional.recall(pred, target, task="multiclass", num_classes=self.num_classes)
    
class F1Metric(_BaseMetrics):
    method = "F1"
    def __init__(
        self,
       num_classes: int,
    ):
        super().__init__()
        assert num_classes > 1, "Number of classes must be greater than 1."
        self.num_classes = num_classes
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_classes == 2:
            return torchmetrics.functional.f1_score(pred, target, task="binary", num_classes=self.num_classes)
        else:
            return torchmetrics.functional.f1_score(pred, target, task="multiclass", num_classes=self.num_classes)

class MetricConfig(BaseModel):
    target_name: str
    """Name of the target metric to monitor."""
    metric_module: _BaseMetrics
    """Method to calculate the metric."""
    normalize_by_num_atoms: bool = False
    """Whether to normalize the metric by the number of atoms."""

class MonitorConfig(BaseModel):
    metrics: list[MetricConfig]
    """Metrics to monitor."""
    primary_metric: MetricConfig
    """Primary metric for early stopping."""
    early_stopping: bool = True
    """Whether to use early stopping."""
    patience: int = 100
    """Number of epochs to wait before early stopping."""
    min_delta: float = 0.0
    """Minimum change in monitored quantity to qualify as improvement."""
    mode: Literal["min", "max"] = "min"
    """In 'min' mode, training will stop when the quantity monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has stopped increasing."""
    logger: _BaseLogger
    """Logger for logging the metrics."""
    
    def construct_monitor(self):
        return Monitor(self)

class Monitor(Generic[TBatch]):
    """Monitor for early stopping and logging."""
    def __init__(self, config: MonitorConfig):
        self.metrics = config.metrics
        self.primary_metric = config.primary_metric
        ## check if primary metric is in metrics, if not, add it
        exist_check = False
        for metric in self.metrics:
            if metric.target_name == self.primary_metric.target_name and \
                metric.metric_module.method == self.primary_metric.metric_module.method and \
                metric.normalize_by_num_atoms == self.primary_metric.normalize_by_num_atoms:
                exist_check = True
                break
        if not exist_check:
            self.metrics.append(self.primary_metric)
        self.early_stopping = config.early_stopping
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.mode = config.mode
        self.logger = config.logger
        self.best_metric = None
        self.wait = 0
        self.perform_early_stop = False
        self.train_metric_cache: dict[str, list[float]] = {}
        self.val_metric_cache: dict[str, list[float]] = {}
        self.test_metric_cache: dict[str, list[float]] = {}
        self.train_loss_cache: dict[str, list[float]] = {}
        self.val_loss_cache: dict[str, list[float]] = {}
        self.test_loss_cache: dict[str, list[float]] = {}
    
    def evaluate(
        self,
        batch: TBatch,
        output_head_results: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Evaluate the metrics."""
        metric_results = {}
        for metric in self.metrics:
            pred = output_head_results[metric.target_name]
            if not hasattr(batch, metric.target_name):
                raise ValueError(f"Target {metric.target_name} not found in batch.")
            target = getattr(batch, metric.target_name)
            assert type(target) == torch.Tensor, "Target must be a tensor."
            assert pred.shape == target.shape, "Prediction and target shapes must match."
            if metric.normalize_by_num_atoms:
                num_atoms = batch.num_atoms.reshape(-1, 1)
                assert num_atoms.shape[0] == pred.shape[0], "Number of atoms must match the batch size"
                pred = pred / num_atoms
                target = target / num_atoms
            metric_result = metric.metric_module(pred, target)
            # print(f"{metric.target_name}-{metric.metric_module.method}: {metric_result}")
            metric_results[f"{metric.target_name}-{metric.metric_module.method}"] = metric_result.item()
        return metric_results
        
    def log_losses_step_end(
        self,
        batch: TBatch,
        loss_results: dict[str, float],
        step_type: Literal["train", "val", "test"],
    ):
        match step_type:
            case "train":
                loss_cache = self.train_loss_cache
            case "val":
                loss_cache = self.val_loss_cache
            case "test":
                loss_cache = self.test_loss_cache
        batch_idx = batch.batch
        num_data = int(torch.max(batch_idx).item()) + 1
        for key, value in loss_results.items():
            if key not in loss_cache:
                loss_cache[key] = [value]*num_data
            else:
                loss_cache[key].extend([value]*num_data)

    def log_losses_epoch_end(
        self,
        epoch_type: Literal["train", "val", "test"],
        current_epoch: int
    ):
        match epoch_type:
            case "train":
                loss_cache = self.train_loss_cache
            case "val":
                loss_cache = self.val_loss_cache
            case "test":
                loss_cache = self.test_loss_cache
        loss_results_epoch = {}
        for key, value in loss_cache.items():
            loss_results_epoch[epoch_type+"/"+key] = sum(value) / len(value)
        self.logger.log(loss_results_epoch, current_epoch)
        loss_cache.clear()
                
        
    def log_metrics_step_end(
        self,
        batch: TBatch,
        metrics_results: dict[str, float],
        step_type: Literal["train", "val", "test"],
    ):
        match step_type:
            case "train":
                metric_cache = self.train_metric_cache
            case "val":
                metric_cache = self.val_metric_cache
            case "test":
                metric_cache = self.test_metric_cache
        batch_idx = batch.batch
        num_data = int(torch.max(batch_idx).item()) + 1
        for key, value in metrics_results.items():
            if key not in metric_cache:
                metric_cache[key] = [value]*num_data
            else:
                metric_cache[key].extend([value]*num_data)
    
    def log_metrics_epoch_end(
        self, 
        epoch_type: Literal["train", "val", "test"], 
        current_epoch: int
    ):
        match epoch_type:
            case "train":
                metric_cache = self.train_metric_cache
            case "val":
                metric_cache = self.val_metric_cache
            case "test":
                metric_cache = self.test_metric_cache
        metrics_results_epoch = {}
        for key, value in metric_cache.items():
            metrics_results_epoch[epoch_type+"/"+key] = sum(value) / len(value)
        self.logger.log(metrics_results_epoch, current_epoch)
        metric_cache.clear()
        
        if epoch_type == "val":
            if self.best_metric is None:
                self.best_metric = metrics_results_epoch[self.primary_metric.target_name]
            else:
                if self.mode == "min":
                    if self.best_metric - metrics_results_epoch[self.primary_metric.target_name] > self.min_delta:
                        self.best_metric = metrics_results_epoch[self.primary_metric.target_name]
                        self.wait = 0
                    else:
                        self.wait += 1
                else:
                    if metrics_results_epoch[self.primary_metric.target_name] - self.best_metric > self.min_delta:
                        self.best_metric = metrics_results_epoch[self.primary_metric.target_name]
                        self.wait = 0
                    else:
                        self.wait += 1
            if self.early_stopping and self.wait >= self.patience:
                self.perform_early_stop = True