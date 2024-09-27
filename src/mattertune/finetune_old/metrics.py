from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Generic, Annotated, Literal, TypeAlias
from typing_extensions import override
from dataclasses import dataclass
from collections import Counter
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import torchmetrics
import jaxtyping as jt
from mattertune.protocol import TBatch, OutputHeadBaseConfig


class _BaseMetrics(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torchmetrics.Metric]: ...


class _BaseMetricsConfig(BaseModel, ABC):
    @abstractmethod
    def create_metrics(self) -> _BaseMetrics: ...


class ScalarRegressionMetricsConfig(_BaseMetricsConfig):
    name: Literal["scalar"] = "scalar"

    mae: bool = True
    """Whether to compute Mean Absolute Error (MAE)"""

    mse: bool = True
    """Whether to compute Mean Squared Error (MSE)"""

    rmse: bool = True
    """Whether to compute Root Mean Squared Error (RMSE)"""

    @override
    def create_metrics(self):
        return ScalarRegressionMetrics(self)


class ScalarRegressionMetrics(_BaseMetrics):
    @override
    def __init__(self, config: ScalarRegressionMetricsConfig):
        super().__init__()

        self.config = config
        del config

        if self.config.mae:
            self.mae = torchmetrics.MeanAbsoluteError()
        if self.config.mse:
            self.mse = torchmetrics.MeanSquaredError(squared=True)
        if self.config.rmse:
            self.rmse = torchmetrics.MeanSquaredError(squared=False)

    @override
    def forward(
        self,
        predictions: jt.Float[torch.Tensor, "..."],
        targets: jt.Float[torch.Tensor, "..."],
    ) -> dict[str, torchmetrics.Metric]:
        metrics: dict[str, torchmetrics.Metric] = {}
        if self.config.mae:
            self.mae(predictions, targets)
            metrics["mae"] = self.mae

        if self.config.mse:
            self.mse(predictions, targets)
            metrics["mse"] = self.mse

        if self.config.rmse:
            self.rmse(predictions, targets)
            metrics["rmse"] = self.rmse

        return metrics


class VectorRegressionMetricsConfig(ScalarRegressionMetricsConfig):
    name: Literal["vector"] = "vector"  # pyright: ignore[reportIncompatibleVariableOverride]

    cos: bool = True
    """Whether to compute Cosine Similarity"""

    @override
    def create_metrics(self):
        return VectorRegressionMetrics(self)


class VectorRegressionMetrics(ScalarRegressionMetrics):
    @override
    def __init__(self, config: VectorRegressionMetricsConfig):
        super().__init__(config)

        self.config = config
        del config

        if self.config.cos:
            self.cos = torchmetrics.CosineSimilarity()

    @override
    def forward(
        self,
        predictions: jt.Float[torch.Tensor, "... d"],
        targets: jt.Float[torch.Tensor, "... d"],
    ) -> dict[str, torchmetrics.Metric]:
        metrics = super().forward(predictions, targets)

        if self.config.cos:
            self.cos(predictions, targets)
            metrics["cos"] = self.cos

        return metrics
    

class TensorRegressionMetricsConfig(ScalarRegressionMetricsConfig):
    name: Literal["vector"] = "vector"  # pyright: ignore[reportIncompatibleVariableOverride]

    cos: bool = True
    """Whether to compute Cosine Similarity"""

    @override
    def create_metrics(self):
        return TensorRegressionMetrics(self)


class TensorRegressionMetrics(ScalarRegressionMetrics):
    @override
    def __init__(self, config: TensorRegressionMetricsConfig):
        super().__init__(config)

        self.config = config
        del config

        if self.config.cos:
            self.cos = torchmetrics.CosineSimilarity()

    @override
    def forward(
        self,
        predictions: jt.Float[torch.Tensor, "... d"],
        targets: jt.Float[torch.Tensor, "... d"],
    ) -> dict[str, torchmetrics.Metric]:
        metrics = super().forward(predictions, targets)

        # if self.config.cos:
        #     self.cos(predictions, targets)
        #     metrics["cos"] = self.cos

        return metrics


class BinaryClassificationMetricsConfig(_BaseMetricsConfig):
    name: Literal["binary_classification"] = "binary_classification"

    accuracy: bool = True
    """Whether to compute accuracy"""

    precision: bool = True
    """Whether to compute precision"""

    recall: bool = True
    """Whether to compute recall"""

    f1: bool = True
    """Whether to compute F1 score"""

    auc: bool = True
    """Whether to compute AUC"""

    ap: bool = True
    """Whether to compute Average Precision"""

    mcc: bool = True
    """Whether to compute Matthews Correlation Coefficient"""

    @override
    def create_metrics(self):
        return BinaryClassificationMetrics(self)


class BinaryClassificationMetrics(_BaseMetrics):
    @override
    def __init__(self, config: BinaryClassificationMetricsConfig):
        super().__init__()

        self.config = config
        del config

        if self.config.accuracy:
            self.accuracy = torchmetrics.Accuracy(task="binary")
        if self.config.precision:
            self.precision = torchmetrics.Precision(task="binary")
        if self.config.recall:
            self.recall = torchmetrics.Recall(task="binary")
        if self.config.f1:
            self.f1 = torchmetrics.F1Score(task="binary")
        if self.config.auc:
            self.auc = torchmetrics.AUROC(task="binary")
        if self.config.ap:
            self.ap = torchmetrics.AveragePrecision(task="binary")
        if self.config.mcc:
            self.mcc = torchmetrics.MatthewsCorrCoef(task="binary")

    @override
    def forward(
        self,
        predictions: jt.Float[torch.Tensor, "..."],
        targets: jt.Int[torch.Tensor, "..."] | jt.Float[torch.Tensor, "..."],
    ) -> dict[str, torchmetrics.Metric]:
        metrics: dict[str, torchmetrics.Metric] = {}
        if self.config.accuracy:
            self.accuracy(predictions, targets)
            metrics["accuracy"] = self.accuracy

        if self.config.precision:
            self.precision(predictions, targets)
            metrics["precision"] = self.precision

        if self.config.recall:
            self.recall(predictions, targets)
            metrics["recall"] = self.recall

        if self.config.f1:
            self.f1(predictions, targets)
            metrics["f1"] = self.f1

        if self.config.auc:
            self.auc(predictions, targets)
            metrics["auc"] = self.auc

        if self.config.ap:
            self.ap(predictions, targets)
            metrics["ap"] = self.ap

        if self.config.mcc:
            self.mcc(predictions, targets)
            metrics["mcc"] = self.mcc

        return metrics


class MulticlassClassificationMetricsConfig(_BaseMetricsConfig):
    name: Literal["multiclass_classification"] = "multiclass_classification"

    accuracy: bool = True
    """Whether to compute accuracy"""

    precision: bool = True
    """Whether to compute precision"""

    recall: bool = True
    """Whether to compute recall"""

    f1: bool = True
    """Whether to compute F1 score"""

    auc: bool = True
    """Whether to compute AUC"""

    ap: bool = True
    """Whether to compute Average Precision"""

    mcc: bool = True
    """Whether to compute Matthews Correlation Coefficient"""

    @override
    def create_metrics(self):
        return MulticlassClassificationMetrics(self)


class MulticlassClassificationMetrics(_BaseMetrics):
    @override
    def __init__(self, config: MulticlassClassificationMetricsConfig):
        super().__init__()

        self.config = config
        del config

        if self.config.accuracy:
            self.accuracy = torchmetrics.Accuracy(task="multiclass")
        if self.config.precision:
            self.precision = torchmetrics.Precision(task="multiclass")
        if self.config.recall:
            self.recall = torchmetrics.Recall(task="multiclass")
        if self.config.f1:
            self.f1 = torchmetrics.F1Score(task="multiclass")
        if self.config.auc:
            self.auc = torchmetrics.AUROC(task="multiclass")
        if self.config.ap:
            self.ap = torchmetrics.AveragePrecision(task="multiclass")
        if self.config.mcc:
            self.mcc = torchmetrics.MatthewsCorrCoef(task="multiclass")

    @override
    def forward(
        self,
        predictions: jt.Float[torch.Tensor, "... c"],
        targets: jt.Int[torch.Tensor, "..."],
    ) -> dict[str, torchmetrics.Metric]:
        metrics: dict[str, torchmetrics.Metric] = {}
        if self.config.accuracy:
            self.accuracy(predictions, targets)
            metrics["accuracy"] = self.accuracy

        if self.config.precision:
            self.precision(predictions, targets)
            metrics["precision"] = self.precision

        if self.config.recall:
            self.recall(predictions, targets)
            metrics["recall"] = self.recall

        if self.config.f1:
            self.f1(predictions, targets)
            metrics["f1"] = self.f1

        if self.config.auc:
            self.auc(predictions, targets)
            metrics["auc"] = self.auc

        if self.config.ap:
            self.ap(predictions, targets)
            metrics["ap"] = self.ap

        if self.config.mcc:
            self.mcc(predictions, targets)
            metrics["mcc"] = self.mcc

        return metrics


MetricsConfig: TypeAlias = Annotated[
    ScalarRegressionMetricsConfig
    | VectorRegressionMetricsConfig
    | BinaryClassificationMetricsConfig
    | MulticlassClassificationMetricsConfig,
    Field(discriminator="name"),
]


class MetricModuleConfig(BaseModel):
    report_mae: bool = True
    """Whether to report MAE"""
    report_mse: bool = True
    """Whether to report MSE in addition to MAE"""
    report_rmse: bool = True
    """Whether to report RMSE in addition to MAE"""
    report_cos: bool = True
    """Whether to report Cosine Similarity in addition to MAE"""
    report_accuracy: bool = True
    """Whether to compute accuracy"""
    report_precision: bool = True
    """Whether to compute precision"""
    report_recall: bool = True
    """Whether to compute recall"""
    report_f1: bool = True
    """Whether to compute F1 score"""
    report_auc: bool = True
    """Whether to compute AUC"""
    report_ap: bool = True
    """Whether to compute Average Precision"""
    report_mcc: bool = True
    """Whether to compute Matthews Correlation Coefficient"""
    per_atom_scaler_metrics: bool = False
    """Whether to compute metrics per atom"""


@dataclass(frozen=True)
class MetricPair:
    predicted: torch.Tensor
    ground_truth: torch.Tensor


@runtime_checkable
class MetricPairProvider(Protocol, Generic[TBatch]):
    def __call__(
        self, prop: str, batch: TBatch, preds: dict[str, torch.Tensor]
    ) -> MetricPair | None: ...

    
class FinetuneMetricsModule(nn.Module, Generic[TBatch]):
    @override
    def __init__(
        self,
        config: MetricModuleConfig,
        provider: MetricPairProvider,
        targets: list[OutputHeadBaseConfig],
    ):
        super().__init__()

        if not isinstance(provider, MetricPairProvider):
            raise TypeError(
                f"Expected {provider=} to be an instance of {MetricPairProvider=}"
            )
        self.provider = provider

        self.config = config
        self.targets = targets
        
        ## Construct all mae metrics
        self.metrics = {}
        for target in self.targets:
            if target.pred_type == "scalar":
                assert target.is_classification() == False, "Classification target should not have scalar prediction type"
                assert self.config.report_mae or self.config.report_mse or self.config.report_rmse, "At least one of mae, mse, rmse should be True"
                self.metrics[target.target_name] = ScalarRegressionMetrics(ScalarRegressionMetricsConfig(
                    mae=self.config.report_mae, 
                    mse=self.config.report_mse, 
                    rmse=self.config.report_rmse,
                ))
            elif target.pred_type == "vector":
                assert target.is_classification() == False, "Classification target should not have vector prediction type"
                assert self.config.report_mae or self.config.report_mse or self.config.report_rmse or self.config.report_cos, "At least one of mae, mse, rmse, cos should be True"
                self.metrics[target.target_name] = VectorRegressionMetrics(VectorRegressionMetricsConfig(
                    mae=self.config.report_mae, 
                    mse=self.config.report_mse, 
                    rmse=self.config.report_rmse,
                    cos=self.config.report_cos
                ))
            elif target.pred_type == "tensor":
                assert target.is_classification() == False, "Classification target should not have tensor prediction type"
                assert self.config.report_mae or self.config.report_mse or self.config.report_rmse, "At least one of mae, mse, rmse, should be True"
                self.metrics[target.target_name] = TensorRegressionMetrics(TensorRegressionMetricsConfig(
                    mae=self.config.report_mae, 
                    mse=self.config.report_mse, 
                    rmse=self.config.report_rmse
                ))
            elif target.pred_type == "classification":
                if target.get_num_classes() == 2:
                    self.metrics[target.target_name] = BinaryClassificationMetrics(BinaryClassificationMetricsConfig(
                        accuracy=self.config.report_accuracy,
                        precision=self.config.report_precision,
                        recall=self.config.report_recall,
                        f1=self.config.report_f1,
                        auc=self.config.report_auc,
                        ap=self.config.report_ap,
                        mcc=self.config.report_mcc
                    ))
                else:
                    self.metrics[target.target_name] = MulticlassClassificationMetrics(MulticlassClassificationMetricsConfig(
                        accuracy=self.config.report_accuracy,
                        precision=self.config.report_precision,
                        recall=self.config.report_recall,
                        f1=self.config.report_f1,
                        auc=self.config.report_auc,
                        ap=self.config.report_ap,
                        mcc=self.config.report_mcc
                    ))
            
        
    @override
    def forward(self, batch: TBatch, preds: dict[str, torch.Tensor]):
        metrics: dict[str, torchmetrics.Metric] = {}
        
        for key, metric in self.metrics.items():
            if (mp := self.provider(key, batch, preds)) is None:
                continue
            metric(mp.predicted, mp.ground_truth)
            for metric_name, metric_value in metric.items():
                metrics[f"{key}_{metric_name}"] = metric_value
            if self.config.per_atom_scaler_metrics:
                num_atoms = batch.num_atoms
                for metric_name, metric_value in metric.items():
                    if metric_value.shape[0] == num_atoms.shape[0]:
                        metrics[f"{key}_per_atom_{metric_name}"] = metric_value / num_atoms
                    else:
                        pass
                    
        return metrics