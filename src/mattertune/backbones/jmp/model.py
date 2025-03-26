from __future__ import annotations

import contextlib
import importlib.util
import logging
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import nshconfig as C
import nshconfig_extra as CE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase import Atoms
from typing_extensions import final, override

from ...finetune import properties as props
from ...finetune.base import (
    FinetuneModuleBase,
    FinetuneModuleBaseConfig,
    ModelOutput,
    _SkipBatchError,
)
from ...normalization import NormalizationContext
from ...registry import backbone_registry
from ...util import optional_import_error_message
from .util import get_activation_cls
from ...finetune.optimizer import PerParamHparamsDict

if TYPE_CHECKING:
    from torch_geometric.data import Batch, Data  # type: ignore[reportMissingImports] # noqa
    from torch_geometric.data.data import BaseData  # type: ignore[reportMissingImports] # noqa

log = logging.getLogger(__name__)


MODEL_URLS = {
    "jmp-s": "https://jmp-iclr-datasets.s3.amazonaws.com/jmp-s.pt",
    "jmp-l": "https://jmp-iclr-datasets.s3.amazonaws.com/jmp-l.pt",
}
CACHE_DIR = Path(torch.hub.get_dir()) / "jmp_checkpoints"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class CutoffsConfig(C.Config):
    main: float
    aeaint: float
    qint: float
    aint: float

    @classmethod
    def from_constant(cls, value: float):
        return cls(main=value, aeaint=value, qint=value, aint=value)


class MaxNeighborsConfig(C.Config):
    main: int
    aeaint: int
    qint: int
    aint: int

    @classmethod
    def from_goc_base_proportions(cls, max_neighbors: int):
        """
        GOC base proportions:
            max_neighbors: 30
            max_neighbors_qint: 8
            max_neighbors_aeaint: 20
            max_neighbors_aint: 1000
        """
        return cls(
            main=max_neighbors,
            aeaint=int(max_neighbors * 20 / 30),
            qint=int(max_neighbors * 8 / 30),
            aint=int(max_neighbors * 1000 / 30),
        )


class JMPGraphComputerConfig(C.Config):
    pbc: bool
    """Whether to use periodic boundary conditions."""

    cutoffs: CutoffsConfig = CutoffsConfig.from_constant(12.0)
    """The cutoff for the radius graph."""

    max_neighbors: MaxNeighborsConfig = MaxNeighborsConfig.from_goc_base_proportions(30)
    """The maximum number of neighbors for the radius graph."""

    per_graph_radius_graph: bool = False
    """Whether to compute the radius graph per graph."""

    def _to_jmp_graph_computer_config(self):
        with optional_import_error_message("jmp"):
            from jmp.models.gemnet.graph import CutoffsConfig, GraphComputerConfig, MaxNeighborsConfig  # type: ignore[reportMissingImports] # noqa # fmt: skip

        return GraphComputerConfig(
            pbc=self.pbc,
            cutoffs=CutoffsConfig(
                main=self.cutoffs.main,
                aeaint=self.cutoffs.aeaint,
                qint=self.cutoffs.qint,
                aint=self.cutoffs.aint,
            ),
            max_neighbors=MaxNeighborsConfig(
                main=self.max_neighbors.main,
                aeaint=self.max_neighbors.aeaint,
                qint=self.max_neighbors.qint,
                aint=self.max_neighbors.aint,
            ),
            per_graph_radius_graph=self.per_graph_radius_graph,
        )


@backbone_registry.register
class JMPBackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["jmp"] = "jmp"
    """The type of the backbone."""

    pretrained_model: str
    """pretrained model name"""

    graph_computer: JMPGraphComputerConfig
    """The configuration for the graph computer."""

    freeze_backbone: bool = False
    """Whether to freeze the backbone during training."""

    @override
    def create_model(self):
        return JMPBackboneModule(self)

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the jmp module is available
        if importlib.util.find_spec("jmp") is None:
            raise ImportError(
                "The jmp module is not installed. Please install it by running"
                " pip install jmp."
            )

        # Make sure torch-geometric is available
        if importlib.util.find_spec("torch_geometric") is None:
            raise ImportError(
                "The torch-geometric module is not installed. Please install it by running"
                " pip install torch-geometric."
            )


@final
class JMPBackboneModule(FinetuneModuleBase["Data", "Batch", JMPBackboneConfig]):
    @override
    @classmethod
    def hparams_cls(cls):
        return JMPBackboneConfig

    @override
    def requires_disabled_inference_mode(self):
        return False

    def _find_potential_energy_prop_name(self):
        for prop in self.hparams.properties:
            if isinstance(prop, props.EnergyPropertyConfig):
                return prop.name
        raise ValueError("No energy property found in the property list")

    def _create_output_head(self, prop: props.PropertyConfig):
        activation_cls = get_activation_cls(self.backbone.hparams.activation)
        match prop:
            case props.EnergyPropertyConfig():
                with optional_import_error_message("jmp"):
                    from jmp.nn.energy_head import EnergyTargetConfig  # type: ignore[reportMissingImports] # noqa

                return EnergyTargetConfig(
                    max_atomic_number=self.backbone.hparams.num_elements
                ).create_model(
                    self.backbone.hparams.emb_size_atom,
                    self.backbone.hparams.emb_size_edge,
                    activation_cls,
                )
            case props.ForcesPropertyConfig():
                if not prop.conservative:
                    with optional_import_error_message("jmp"):
                        from jmp.nn.force_head import ForceTargetConfig  # type: ignore[reportMissingImports] # noqa

                    return ForceTargetConfig().create_model(
                        self.backbone.hparams.emb_size_edge, activation_cls
                    )
                else:
                    with optional_import_error_message("jmp"):
                        from jmp.nn.force_head import ConservativeForceTargetConfig  # type: ignore[reportMissingImports] # noqa

                    force_config = ConservativeForceTargetConfig(
                        energy_prop_name=self._find_potential_energy_prop_name()
                    )
                    return force_config.create_model()

            case props.StressesPropertyConfig():
                if not prop.conservative:
                    with optional_import_error_message("jmp"):
                        from jmp.nn.stress_head import StressTargetConfig  # type: ignore[reportMissingImports] # noqa

                    return StressTargetConfig().create_model(
                        self.backbone.hparams.emb_size_edge, activation_cls
                    )
                else:
                    with optional_import_error_message("jmp"):
                        from jmp.nn.stress_head import ConservativeStressTargetConfig  # type: ignore[reportMissingImports] # noqa

                    stress_config = ConservativeStressTargetConfig(
                        energy_prop_name=self._find_potential_energy_prop_name()
                    )
                    return stress_config.create_model()
            case props.GraphPropertyConfig():
                with optional_import_error_message("jmp"):
                    from jmp.nn.graph_scaler import GraphScalarTargetConfig  # type: ignore[reportMissingImports] # noqa

                return GraphScalarTargetConfig(reduction=prop.reduction).create_model(
                    self.backbone.hparams.emb_size_atom,
                    activation_cls,
                )
            case _:
                raise ValueError(
                    f"Unsupported property config: {prop} for JMP"
                    "Please ask the maintainers of JMP for support"
                )

    @override
    def create_model(self):
        # Resolve the checkpoint path
        pretrained_model = self.hparams.pretrained_model
        if pretrained_model in MODEL_URLS:
            cached_ckpt_path = CACHE_DIR / f"{pretrained_model}.pt"
            if not cached_ckpt_path.exists():
                log.info(
                    f"Downloading the pretrained model from {MODEL_URLS[pretrained_model]}"
                )
                torch.hub.download_url_to_file(
                    MODEL_URLS[pretrained_model], str(cached_ckpt_path)
                )
            ckpt_path = cached_ckpt_path
        else:
            ckpt_path = None
            raise ValueError(
                f"Unknown pretrained model: {pretrained_model}, available models: {MODEL_URLS.keys()}"
            )

        # Load the backbone from the checkpoint
        with optional_import_error_message("jmp"):
            from jmp.models.gemnet import GemNetOCBackbone  # type: ignore[reportMissingImports] # noqa
            from jmp.models.gemnet.graph import GraphComputer  # type: ignore[reportMissingImports] # noqa

        assert ckpt_path is not None
        self.backbone = GemNetOCBackbone.from_pretrained_ckpt(ckpt_path)
                
        log.info(
            f"Loaded the model from the checkpoint at {ckpt_path}. The model "
            f"has {sum(p.numel() for p in self.backbone.parameters()):,} parameters."
        )

        # Create the graph computer
        self.graph_computer = GraphComputer(
            self.hparams.graph_computer._to_jmp_graph_computer_config(),
            self.backbone.hparams,
        )

        # Create the output heads
        self.output_heads = nn.ModuleDict()
        ## Rearange the properties to move the energy property to the front and stress second
        self.hparams.properties = sorted(
            self.hparams.properties,
            key=lambda prop: (
                not isinstance(prop, props.EnergyPropertyConfig),
                not isinstance(prop, props.StressesPropertyConfig),
            ),
        )
        for prop in self.hparams.properties:
            self.output_heads[prop.name] = self._create_output_head(prop)

    @override
    def trainable_parameters(self):
        if not self.hparams.freeze_backbone:
            yield from self.backbone.named_parameters()
        for head in self.output_heads.values():
            yield from head.named_parameters()

    @override
    @contextlib.contextmanager
    def model_forward_context(self, data, mode: str):
        with ExitStack() as stack:
            for head in self.output_heads.values():
                stack.enter_context(head.forward_context(data))

            yield

    @override
    def model_forward(self, batch, mode: str, return_backbone_output=False, using_partition=False):
        # Run the backbone
        if return_backbone_output:
            backbone_output, intermediate = self.backbone(
                batch, return_intermediate=True
            )
        else:
            backbone_output = self.backbone(batch)

        # Feed the backbone output to the output heads
        predicted_properties: dict[str, torch.Tensor] = {}

        head_input: dict[str, Any] = {
            "data": batch,
            "backbone_output": backbone_output,
            "predicted_props": predicted_properties,
        }
        for name, head in self.output_heads.items():
            assert (
                prop := next(
                    (p for p in self.hparams.properties if p.name == name), None
                )
            ) is not None, (
                f"Property {name} not found in properties. "
                "This should not happen, please report this."
            )
            if using_partition and isinstance(prop, props.EnergyPropertyConfig):
                output, per_atom_energies = head(head_input, return_per_atom_energy=True)
                head_input["predicted_props"]["energies_per_atom"] = per_atom_energies
            else:
                output = head(head_input)
            if torch.isnan(output).any() or torch.isinf(output).any():
                raise _SkipBatchError("NaN or inf detected in the output")
            head_input["predicted_props"][name] = output

        pred: ModelOutput = {"predicted_properties": predicted_properties}
        if return_backbone_output:
            pred["backbone_output"] = intermediate # type: ignore[assignment]
        return pred

    @override
    def apply_callable_to_backbone(self, fn):
        return fn(self.backbone)

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
        with optional_import_error_message("torch_geometric"):
            from torch_geometric.data import Batch  # type: ignore[reportMissingImports] # noqa

        return Batch.from_data_list(cast("list[BaseData]", data_list))

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
        with optional_import_error_message("torch_geometric"):
            from torch_geometric.data import Data  # type: ignore[reportMissingImports] # noqa

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
    
    @override
    def get_connectivity_from_data(self, data) -> torch.Tensor:
        graph = self.graph_computer(data)
        edge_indices = graph["main_edge_index"].clone()
        return edge_indices
    
    @override
    def get_connectivity_from_atoms(self, atoms) -> torch.Tensor:
        data = self.atoms_to_data(atoms, has_labels=False)
        return self.get_connectivity_from_data(data)

    @override
    def create_normalization_context_from_batch(self, batch):
        with optional_import_error_message("torch_scatter"):
            from torch_scatter import scatter  # type: ignore[reportMissingImports] # noqa

        atomic_numbers: torch.Tensor = batch["atomic_numbers"].long()  # (n_atoms,)
        batch_idx: torch.Tensor = batch["batch"]  # (n_atoms,)
        
        ## get num_atoms per sample
        all_ones = torch.ones_like(atomic_numbers)
        num_atoms = scatter(
            all_ones,
            batch_idx,
            dim=0,
            dim_size=batch.num_graphs,
            reduce="sum",
        )

        # Convert atomic numbers to one-hot encoding
        atom_types_onehot = F.one_hot(atomic_numbers, num_classes=120)

        compositions = scatter(
            atom_types_onehot,
            batch_idx,
            dim=0,
            dim_size=batch.num_graphs,
            reduce="sum",
        )
        compositions = compositions[:, 1:]  # Remove the zeroth element
        return NormalizationContext(num_atoms=num_atoms, compositions=compositions)
    
    @override
    def apply_early_stop_message_passing(self, message_passing_steps: int|None):
        """
        Apply message passing for early stopping.
        """
        if message_passing_steps is None:
            pass
        else:
            self.backbone.num_blocks = min(self.backbone.num_blocks, message_passing_steps)


def _get_fixed(atoms: Atoms):
    """Gets the fixed atom constraint mask from an Atoms object."""
    fixed = np.zeros(len(atoms), dtype=np.bool_)
    if (constraints := getattr(atoms, "constraints", None)) is None:
        raise ValueError("Atoms object does not have a constraints attribute")

    from ase.constraints import FixAtoms

    for constraint in constraints:
        if not isinstance(constraint, FixAtoms):
            continue
        fixed[constraint.index] = True

    return fixed


def get_jmp_s_lr_decay(lr: float):
    per_parameter_hparams = [
        {
            "patterns": ["embedding.*"],
            "hparams": {
                "lr": 0.3 * lr,
            },
        },
        {
            "patterns": ["int_blocks.0.*"],
            "hparams": {
                "lr": 0.3 * lr,
            },
        },
        {
            "patterns": ["int_blocks.1.*"],
            "hparams": {
                "lr": 0.4 * lr,
            },
        },
        {
            "patterns": ["int_blocks.2.*"],
            "hparams": {
                "lr": 0.55 * lr,
            },
        },
        {
            "patterns": ["int_blocks.3.*"],
            "hparams": {
                "lr": 0.625 * lr,
            },
        },
    ]
    return cast(Sequence[PerParamHparamsDict], per_parameter_hparams)

def get_jmp_l_lr_decay(lr: float):
    per_parameter_hparams = [
        {
            "patterns": ["embedding.*"],
            "hparams": {
                "lr": 0.3 * lr,
            },
        },
        {
            "patterns": ["int_blocks.0.*"],
            "hparams": {
                "lr": 0.55 * lr,
            },
        },
        {
            "patterns": ["int_blocks.1.*"],
            "hparams": {
                "lr": 0.4 * lr,
            },
        },
        {
            "patterns": ["int_blocks.2.*"],
            "hparams": {
                "lr": 0.3 * lr,
            },
        },
        {
            "patterns": ["int_blocks.3.*"],
            "hparams": {
                "lr": 0.4 * lr,
            },
        },
        {
            "patterns": ["int_blocks.4.*"],
            "hparams": {
                "lr": 0.55 * lr,
            },
        },
        {
            "patterns": ["int_blocks.5.*"],
            "hparams": {
                "lr": 0.625 * lr,
            },
        },
    ]
    return cast(Sequence[PerParamHparamsDict], per_parameter_hparams)