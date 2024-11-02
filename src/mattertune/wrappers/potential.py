from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch import Trainer
from torch.utils.data import Dataset
from typing_extensions import override

from ..finetune.properties import PropertyConfig

if TYPE_CHECKING:
    from ase import Atoms

    from ..finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig


class MatterTunePotential:
    def __init__(
        self,
        lightning_module: FinetuneModuleBase[Any, Any, FinetuneModuleBaseConfig],
        lightning_trainer_kwargs: dict[str, Any] | None = None,
    ):
        self.lightning_module = lightning_module
        self._lightning_trainer_kwargs = lightning_trainer_kwargs or {}

    def predict(
        self,
        atoms_list: list[Atoms],
        properties: Sequence[str | PropertyConfig],
    ) -> list[dict[str, torch.Tensor]]:
        # Resolve `properties` to a list of `PropertyConfig` objects.
        properties = _resolve_properties(properties, self.lightning_module.hparams)

        # Create a trainer instance.
        trainer = _create_trainer(self._lightning_trainer_kwargs)

        # Create a dataloader from the atoms_list.
        dataloader = _atoms_list_to_dataloader(atoms_list, self.lightning_module)
        predictions = trainer.predict(
            self.lightning_module, dataloader, return_predictions=True
        )
        assert predictions is not None, "Predictions should not be None. Report a bug."

        return predictions


def _resolve_properties(
    properties: Sequence[str | PropertyConfig],
    hparams: FinetuneModuleBaseConfig,
):
    resolved_properties: list[PropertyConfig] = []
    for prop in properties:
        # If `PropertyConfig`, append it to the list.
        if not isinstance(prop, str):
            resolved_properties.append(prop)
            continue

        # If string, it must be present in the hparams.
        if (
            prop := next((p for p in hparams.properties if p.name == prop), None)
        ) is None:
            raise ValueError(
                f"Property '{prop}' not found in the model hyperparameters. "
                f"Available properties: {', '.join(p.name for p in hparams.properties)}"
            )
        resolved_properties.append(prop)

    return resolved_properties


def _create_trainer(trainer_kwargs: dict[str, Any]):
    return Trainer(**trainer_kwargs)


def _atoms_list_to_dataloader(
    atoms_list: list[Atoms],
    lightning_module: FinetuneModuleBase,
):
    class AtomsDataset(Dataset):
        def __init__(self, atoms_list: list[Atoms]):
            self.atoms_list = atoms_list

        def __len__(self):
            return len(self.atoms_list)

        @override
        def __getitem__(self, idx):
            return self.atoms_list[idx]

    dataset = AtomsDataset(atoms_list)
    dataloader = lightning_module.create_dataloader(
        dataset,
        has_labels=False,
        batch_size=1,
    )
    return dataloader
