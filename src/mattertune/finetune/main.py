from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import nshconfig as C
from lightning.pytorch import Trainer

from ..backbones import BackboneConfig
from ..data import DatasetConfig

if TYPE_CHECKING:
    from ..data.loader import DataLoaderKwargs


class PerSplitDataConfig(C.Config):
    train: DatasetConfig
    """The configuration for the training data."""

    validation: DatasetConfig | None = None
    """The configuration for the validation data."""

    batch_size: int
    """The batch size for the dataloaders.
    TODO: Add support for auto batch size tuning."""

    num_workers: int | Literal["auto"] = "auto"
    """The number of workers for the dataloaders.

    This is the number of processes that generate batches in parallel.
    If set to "auto", the number of workers will be automatically
        set based on the number of available CPUs.
    Set to 0 to disable parallelism.
    """

    pin_memory: bool = True
    """Whether to pin memory in the dataloaders.

    This is useful for speeding up GPU data transfer.
    """

    def _num_workers_or_auto(self):
        if self.num_workers == "auto":
            import os

            if (cpu_count := os.cpu_count()) is not None:
                return cpu_count - 1
            else:
                return 1

        return self.num_workers

    def dataloader_kwargs(self) -> DataLoaderKwargs:
        return {
            "batch_size": self.batch_size,
            "num_workers": self._num_workers_or_auto(),
            "pin_memory": self.pin_memory,
        }


class MatterTunerConfig(C.Config):
    data: PerSplitDataConfig
    """The configuration for the data."""

    model: BackboneConfig
    """The configuration for the model."""

    lightning_trainer_kwargs: dict[str, Any] = {}
    """
    Keyword arguments for the Lightning Trainer.
    This is for advanced users who want to customize the Lightning Trainer,
        and is not recommended for beginners.
    """


class MatterTuner:
    def __init__(self, config: MatterTunerConfig):
        self.config = config

    def tune(self):
        # Resolve the model class
        model_cls = self.config.model.model_cls()

        # Create the model
        lightning_module = model_cls(self.config.model)
        assert isinstance(
            lightning_module, FinetuneModuleBase
        ), f'The backbone model must be a FinetuneModuleBase subclass. Got "{type(lightning_module)}".'

        # Create the datasets & wrap them in our dataloader logic
        train_dataloader = lightning_module.create_dataloader(
            self.config.data.train.create_dataset(),
            has_labels=True,
            **self.config.data.dataloader_kwargs(),
        )
        val_dataloader = (
            lightning_module.create_dataloader(
                self.config.data.validation.create_dataset(),
                has_labels=True,
                **self.config.data.dataloader_kwargs(),
            )
            if self.config.data.validation is not None
            else None
        )

        # Create the trainer
        trainer = Trainer(**self.config.lightning_trainer_kwargs)
        trainer.fit(
            lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # Return the trained model
        return lightning_module
