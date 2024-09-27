from abc import ABC, abstractmethod
from typing import Annotated, Literal, TypeAlias

import nshconfig as C
import nshutils.typecheck as tc
import torch
import torch.nn as nn
import torchmetrics
from typing_extensions import override


class _BaseMetrics(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torchmetrics.Metric]: ...


class _BaseMetricsConfig(C.Config, ABC):
    @abstractmethod
    def create_metrics(self) -> _BaseMetrics: ...


class ScalarRegressionMetricsConfig(_BaseMetricsConfig):
    type: Literal["scalar"] = "scalar"

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
        predictions: tc.Float[torch.Tensor, "..."],
        targets: tc.Float[torch.Tensor, "..."],
    ):
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
    type: Literal["vector"] = "vector"  # pyright: ignore[reportIncompatibleVariableOverride]

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
        predictions: tc.Float[torch.Tensor, "... d"],
        targets: tc.Float[torch.Tensor, "... d"],
    ):
        metrics = super().forward(predictions, targets)

        if self.config.cos:
            self.cos(predictions, targets)
            metrics["cos"] = self.cos

        return metrics


class BinaryClassificationMetricsConfig(_BaseMetricsConfig):
    type: Literal["binary_classification"] = "binary_classification"

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
        predictions: tc.Float[torch.Tensor, "..."],
        targets: tc.Int[torch.Tensor, "..."] | tc.Float[torch.Tensor, "..."],
    ):
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
    type: Literal["multiclass_classification"] = "multiclass_classification"

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
        predictions: tc.Float[torch.Tensor, "... c"],
        targets: tc.Int[torch.Tensor, "..."],
    ):
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
    C.Discriminator("type"),
]