from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeAlias, Annotated
from pydantic import BaseModel, Field
from mattertune.data_structures import TMatterTuneBatch
import torch
import torch.nn as nn
import torchmetrics

class _BaseMetrics(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor: ...
    
class MAEMetric(_BaseMetrics):
    name: Literal["MAE"] = "MAE"
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torchmetrics.functional.mean_absolute_error(pred, target)
    
class RMSEMetric(_BaseMetrics):
    name: Literal["RMSE"] = "RMSE"
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torchmetrics.functional.mean_squared_error(pred, target).sqrt()

class CosineSimilarityMetric(_BaseMetrics):
    name: Literal["CosineSimilarity"] = "CosineSimilarity"
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return torchmetrics.functional.cosine_similarity(pred, target)

class AccuracyMetric(_BaseMetrics):
    name: Literal["Accuracy"] = "Accuracy"
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
    name: Literal["Precision"] = "Precision"
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
    name:Literal["Recall"] = "Recall"
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
    name: Literal["F1"] = "F1"
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
        
        
# MetricCalculator: TypeAlias = Annotated[
#     MAEMetric | RMSEMetric | CosineSimilarityMetric | AccuracyMetric | PrecisionMetric | RecallMetric | F1Metric,
#     Field(discriminator="name"),
# ]


class MetricConfig(BaseModel):
    model_config = {
        'arbitrary_types_allowed': True
    }
    target_name: str
    """Name of the target metric to monitor."""
    metric_calculator: _BaseMetrics
    """Method to calculate the metric."""
    normalize_by_num_atoms: bool = False
    """Whether to normalize the metric by the number of atoms."""
    
    def get_metric_name(self):
        name = f"{self.target_name}-{self.metric_calculator.name}"
        if self.normalize_by_num_atoms:
            name += "-peratom"
        return name
    

class MetricsModuleConfig(BaseModel):
    """
    Configuration for the metrics module.
    Defined by a list of MetricConfig objects.
    Also deals with early stopping based on one primary metric.
    """
    metrics: list[MetricConfig]
    """List of MetricConfig objects."""
    primary_metric: MetricConfig
    """Primary metric to use for early stopping."""
    
    def construct_metrics_module(self):
        return MetricsModule(self)
    
    
class MetricsModule(Generic[TMatterTuneBatch]):
    """
    Module to calculate metrics.
    """
    def __init__(
        self,
        config: MetricsModuleConfig,
    ):
        self.metrics = config.metrics
        self.primary_metric = config.primary_metric
        ## check if primary metric is in metrics, if not, add it
        exist_check = False
        for metric in self.metrics:
            if metric.get_metric_name() == self.primary_metric.get_metric_name():
                break
        if not exist_check:
            self.metrics.append(self.primary_metric)
            
    def get_primary_metric_name(self):
        return self.primary_metric.get_metric_name()
    
    def compute(
        self,
        batch: TMatterTuneBatch,
        output_head_results: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """
        Compute the metrics.
        """
        batch_labels: dict[str, torch.Tensor] = batch.labels
        metric_results = {}
        for metric in self.metrics:
            pred = output_head_results[metric.target_name]
            target = batch_labels[metric.target_name]
            assert pred.shape == target.shape, f"Prediction and target shapes must match. For {metric.target_name}, got {pred.shape} and {target.shape}"
            if metric.normalize_by_num_atoms:
                num_atoms = batch.num_atoms.reshape(-1, 1)
                assert num_atoms.shape[0] == pred.shape[0], "Number of atoms must match the batch size"
                pred = pred / num_atoms
                target = target / num_atoms
            metric_name = metric.get_metric_name()
            metric_results[metric_name] = metric.metric_calculator(pred, target).detach().cpu().item()
        return metric_results