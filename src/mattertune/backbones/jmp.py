from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import nshconfig_extra as CE
import numpy as np
import torch
import torch.nn as nn
from jmp.models.gemnet.graph import GraphComputer, GraphComputerConfig
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from typing_extensions import override

from ..finetune import properties as props
from ..finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig, ModelOutput

if TYPE_CHECKING:
    from ase import Atoms

log = logging.getLogger(__name__)


def _get_activation_cls(activation: str) -> type[nn.Module]:
    """
    Get the activation class from the activation name
    """
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU
    elif activation == "silu" or activation == "swish":
        return nn.SiLU
    elif activation == "scaled_silu" or activation == "scaled_swish":
        from jmp.models.gemnet.layers.base_layers import ScaledSiLU

        return ScaledSiLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "identity":
        return nn.Identity
    else:
        raise ValueError(f"Activation {activation} is not supported")


class JMPBackboneConfig(FinetuneModuleBaseConfig):
    type: Literal["jmp"] = "jmp"
    """The type of the backbone."""

    ckpt_path: Path | CE.CachedPath
    """The path to the pre-trained model checkpoint."""

    graph_computer: GraphComputerConfig
    """The configuration for the graph computer."""

    @override
    def create_backbone(self):
        return JMPBackboneModule(self)


class JMPBackboneModule(FinetuneModuleBase[Data, Batch, JMPBackboneConfig]):
    def _create_output_head(self, prop: props.PropertyConfig):
        activation_cls = _get_activation_cls(self.backbone.hparams.activation)
        match prop:
            case props.EnergyPropertyConfig():
                from jmp.nn.energy_head import EnergyTargetConfig

                return EnergyTargetConfig(
                    max_atomic_number=self.backbone.hparams.num_elements
                ).create_model(
                    self.backbone.hparams.emb_size_atom,
                    self.backbone.hparams.emb_size_edge,
                    activation_cls,
                )
            case props.ForcesPropertyConfig(conservative=False):
                from jmp.nn.force_head import ForceTargetConfig

                return ForceTargetConfig().create_model(
                    self.backbone.hparams.emb_size_edge, activation_cls
                )
            case props.StressesPropertyConfig(conservative=False):
                from jmp.nn.stress_head import StressTargetConfig

                return StressTargetConfig().create_model(
                    self.backbone.hparams.emb_size_edge, activation_cls
                )
            case _:
                raise ValueError(
                    f"Unsupported property config: {prop}. "
                    "Please ask the maintainers of the JMP backbone to add support for it."
                )

    @override
    def create_model(self):
        # Resolve the checkpoint path
        if isinstance(ckpt_path := self.hparams.ckpt_path, CE.CachedPath):
            ckpt_path = ckpt_path.resolve()

        # Load the backbone from the checkpoint
        from jmp.models.gemnet import GemNetOCBackbone

        self.backbone = GemNetOCBackbone.from_pretrained_ckpt(ckpt_path)
        log.info(
            f"Loaded the model from the checkpoint at {ckpt_path}. The model "
            f"has {sum(p.numel() for p in self.backbone.parameters()):,} parameters."
        )

        # Create the graph computer
        self.graph_computer = GraphComputer(
            self.hparams.graph_computer,
            self.backbone.hparams,
        )

        # Create the output heads
        self.output_heads = nn.ModuleDict()
        for prop in self.hparams.properties:
            self.output_heads[prop.name] = self._create_output_head(prop)

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

        head_input = {"data": batch, "backbone_output": backbone_output}
        for name, head in self.output_heads.items():
            predicted_properties[name] = head(head_input)

        pred: ModelOutput = {"predicted_properties": predicted_properties}
        if return_backbone_output:
            pred["backbone_output"] = backbone_output
        return pred

    @override
    def pretrained_backbone_parameters(self):
        return self.backbone.parameters()

    @override
    def output_head_parameters(self):
        for head in self.output_heads.values():
            yield from head.parameters()

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
        for prop in self.hparams.properties:
            labels[prop.name] = getattr(batch, prop.name)
        return labels

    @override
    def atoms_to_data(self, atoms, has_labels):
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
        # - cell: The cell vectors (shape: (1, 3, 3))
        # - pbc: The periodic boundary conditions (shape: (1, 3))
        data_dict: dict[str, torch.Tensor] = {
            "pos": torch.tensor(atoms.positions, dtype=torch.float32),
            "atomic_numbers": torch.tensor(atoms.numbers, dtype=torch.long),
            "natoms": torch.tensor(len(atoms), dtype=torch.long),
            "tags": torch.full((len(atoms),), 2, dtype=torch.long),
            "fixed": torch.from_numpy(_get_fixed(atoms)).bool(),
            "cell": torch.from_numpy(np.array(atoms.cell, dtype=np.float32))
            .float()
            .unsqueeze(0),
            "pbc": torch.tensor(atoms.pbc, dtype=torch.bool).unsqueeze(0),
        }

        if has_labels:
            # Also, pass along any other targets/properties. This includes:
            #   - energy: The total energy of the system
            #   - forces: The forces on each atom
            #   - stress: The stress tensor of the system
            #   - anything else you want to predict
            for prop in self.hparams.properties:
                value = prop._from_ase_atoms_to_torch(atoms)
                # For stress, we should make sure it is (3, 3), not the flattened (6,)
                #   that ASE returns.
                if isinstance(prop, props.StressesPropertyConfig):
                    from ase.constraints import voigt_6_to_full_3x3_stress

                    value = voigt_6_to_full_3x3_stress(value.float().numpy())
                    value = torch.from_numpy(value).float().reshape(1, 3, 3)

                data_dict[prop.name] = value

        return Data.from_dict(data_dict)


def _get_fixed(atoms: Atoms):
    """Gets the fixed atom constraint mask from an Atoms object."""
    fixed = np.zeros(len(atoms), dtype=np.bool_)
    if not hasattr(atoms, "constraints"):
        raise ValueError("Atoms object does not have a constraints attribute")

    from ase.constraints import FixAtoms

    constraint = next((c for c in atoms.constraints if isinstance(c, FixAtoms)), None)
    if constraint is None:
        raise ValueError("Atoms object does not have a FixAtoms constraint")

    fixed[constraint.index] = True
    return fixed
