from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast, final

import nshconfig_extra as CE
import torch
import torch.nn as nn
from jmp.models.gemnet.graph import GraphComputer, GraphComputerConfig
from torch_geometric.data import Batch, Data
from typing_extensions import override

from ..finetune import properties as props
from ..finetune.base import (
    FinetuneModuleBase,
    FinetuneModuleBaseConfig,
    ModelPrediction,
)

if TYPE_CHECKING:
    from torch_geometric.data.data import BaseData

log = logging.getLogger(__name__)


@final
class JMPBackboneConfig(FinetuneModuleBaseConfig):
    ckpt_path: Path | CE.CachedPath
    """The path to the pre-trained model checkpoint."""

    graph_computer: GraphComputerConfig
    """The configuration for the graph computer."""


@final
class JMPBackboneModule(FinetuneModuleBase[Data, Batch, JMPBackboneConfig]):
    @override
    def create_model(self):
        # Resolve the checkpoint path
        if isinstance(ckpt_path := self.hparams.ckpt_path, CE.CachedPath):
            ckpt_path = ckpt_path.resolve()

        # Load the backbone from the checkpoint
        from jmp.models.gemnet import GemNetOCBackbone

        self.backbone = GemNetOCBackbone.from_pretrained_ckpt(ckpt_path)
        log.info(
            f"Loaded the model from the checkpoint at {ckpt_path}. The model has {sum(p.numel() for p in self.backbone.parameters()):,} parameters."
        )

        # Create the graph computer
        self.graph_computer = GraphComputer(
            self.hparams.graph_computer,
            self.backbone.hparams,
        )

        # Create the output heads
        raise NotImplementedError("Implement this!")
        self.output_heads = nn.ModuleDict()
        for name, config in self.hparams.properties.items():
            match config:
                case props.GraphPropertyConfig():
                    pass
                case _:
                    raise ValueError(
                        f"Unsupported property config: {config}. "
                        "Please ask the maintainers of the JMP backbone to add support for it."
                    )

    @override
    @contextlib.contextmanager
    def model_forward_context(self, data):
        yield

    @override
    def model_forward(self, batch, return_backbone_output=False):
        # Run the backbone
        backbone_output = self.backbone(batch)

        # Feed the backbone output to the output heads
        predicted_properties: dict[str, torch.Tensor] = {}
        for name, head in self.output_heads.items():
            predicted_properties[name] = head(batch, backbone_output)

        pred: ModelPrediction = {"predicted_properties": predicted_properties}
        if return_backbone_output:
            pred["backbone_output"] = backbone_output
        return pred

    @override
    def pretrained_backbone_parameters(self):
        return self.backbone.parameters()

    @override
    def cpu_data_transform(self, data):
        return data

    @override
    def collate_fn(self, data_list):
        return Batch.from_data_list(cast(list[BaseData], data_list))

    @override
    def gpu_batch_transform(self, batch):
        return self.graph_computer(batch)

    @override
    def batch_to_labels(self, batch):
        labels: dict[str, torch.Tensor] = {}
        for name in self.hparams.properties.keys():
            labels[name] = getattr(batch, name)
        return labels

    @override
    def atoms_to_data(self, atoms):
        # For JMP, your PyG object should have the following attributes:
        # - pos: Node positions (shape: (N, 3))
        # - atomic_numbers: Atomic numbers (shape: (N,))
        # - natoms: Number of atoms (shape: (), i.e. a scalar)
        # - tags: Atom tags (shape: (N,)), this is used to distinguish between
        #       surface and adsorbate atoms in datasets like OC20.
        #       Set this to 2 if you don't have this information.
        # - fixed: Boolean tensor indicating whether an atom is fixed
        #       in the relaxation (shape: (N,)), set this to False
        #       if you don't have this information.
        data_dict: dict[str, torch.Tensor] = {
            "pos": torch.tensor(atoms.positions, dtype=torch.float32),
            "atomic_numbers": torch.tensor(atoms.numbers, dtype=torch.long),
            "natoms": torch.tensor(len(atoms), dtype=torch.long),
            "tags": torch.full((len(atoms),), 2, dtype=torch.long),
            "fixed": torch.zeros(len(atoms), dtype=torch.bool),
        }

        # Also, pass along any other targets/properties.
        # This includes:
        # - energy: The total energy of the system
        # - forces: The forces on each atom
        # - stress: The stress tensor of the system
        # - anything else you want to predict
        for name, config in self.hparams.properties.items():
            data_dict[]

        return Data.from_dict(data_dict)
