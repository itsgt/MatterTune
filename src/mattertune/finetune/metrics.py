from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeAlias, Annotated
from pydantic import BaseModel, Field
from mattertune.protocol import TBatch
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
    name : Literal["MAE"] = "MAE"
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
    name: Literal["Recall"] = "Recall"
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
    
    
class MetricsModule(Generic[TBatch]):
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
            if metric.target_name == self.primary_metric.target_name and \
                metric.metric_calculator.name == self.primary_metric.metric_calculator.name and \
                metric.normalize_by_num_atoms == self.primary_metric.normalize_by_num_atoms:
                exist_check = True
                break
        if not exist_check:
            self.metrics.append(self.primary_metric)
    
    def compute(
        self,
        batch: TBatch,
        output_head_results: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """
        Compute the metrics.
        """
        metric_results = {}
        for metric in self.metrics:
            pred = output_head_results[metric.target_name]
            if not hasattr(batch, metric.target_name):
                raise ValueError(f"Batch does not have the target {metric.target_name}.")
            target = getattr(batch, metric.target_name)
            assert isinstance(target, torch.Tensor), "Target must be a tensor."
            assert pred.shape == target.shape, "Prediction and target shapes must match."
            if metric.normalize_by_num_atoms:
                num_atoms = batch.num_atoms.reshape(-1, 1)
                assert num_atoms.shape[0] == pred.shape[0], "Number of atoms must match the batch size"
                pred = pred / num_atoms
                target = target / num_atoms
            metric_name = f"{metric.target_name}-{metric.metric_calculator.name}"
            if metric.normalize_by_num_atoms:
                metric_name += "-peratom"
            metric_results[metric_name] = metric.metric_calculator(pred, target).detach().cpu().item()
        return metric_results