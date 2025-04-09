from __future__ import annotations

import contextlib
import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from functools import partial

import nshconfig as C
import nshconfig_extra as CE
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import assert_never, final, override

from ...finetune import properties as props
from ...finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig, ModelOutput
from ...normalization import NormalizationContext
from ...registry import backbone_registry
from ...util import optional_import_error_message

with optional_import_error_message("fairchem"):
    from fairchem.core.common import gp_utils
    from fairchem.core.models.base import GraphData, HeadInterface
    from fairchem.core.models.equiformer_v2.transformer_block import FeedForwardNetwork
    from fairchem.core.models.equiformer_v2.equiformer_v2 import eqv2_init_weights

if TYPE_CHECKING:
    from torch_geometric.data.batch import Batch  # type: ignore[reportMissingImports] # noqa
    from torch_geometric.data.data import BaseData  # type: ignore[reportMissingImports] # noqa

log = logging.getLogger(__name__)

# https://github.com/FAIR-Chem/fairchem/blob/omat24/src/fairchem/core/models/equiformer_v2/equiformer_v2.py
class EquiformerV2ScalarHead(nn.Module, HeadInterface):
    def __init__(self, backbone, reduce: str = "sum"):
        super().__init__()
        self.reduce = reduce
        self.avg_num_nodes = backbone.avg_num_nodes
        self.energy_block = FeedForwardNetwork(
            backbone.sphere_channels,
            backbone.ffn_hidden_channels,
            1,
            backbone.lmax_list,
            backbone.mmax_list,
            backbone.SO3_grid,
            backbone.ffn_activation,
            backbone.use_gate_act,
            backbone.use_grid_mlp,
            backbone.use_sep_s2_act,
        )
        self.apply(partial(eqv2_init_weights, weight_init=backbone.weight_init))

    def forward(self, data: Batch, emb: dict[str, torch.Tensor | GraphData]):
        node_energy = self.energy_block(emb["node_embedding"])
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        if gp_utils.initialized():
            node_energy = gp_utils.gather_from_model_parallel_region(node_energy, dim=0)
        energy = torch.zeros(
            len(data.natoms),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )

        energy.index_add_(0, data.batch, node_energy.view(-1))
        if self.reduce == "sum":
            return {"energy": energy / self.avg_num_nodes}
        elif self.reduce == "mean":
            return {"energy": energy / data.natoms}
        else:
            raise ValueError(
                f"reduce can only be sum or mean, user provided: {self.reduce}"
            )

class FAIRChemAtomsToGraphSystemConfig(C.Config):
    """Configuration for converting ASE Atoms to a graph for the FAIRChem model."""

    radius: float
    """The radius for edge construction."""
    max_num_neighbors: int
    """The maximum number of neighbours each node can send messages to."""


@backbone_registry.register
class EqV2BackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["eqV2"] = "eqV2"
    """The type of the backbone."""

    checkpoint_path: Path | CE.CachedPath
    """The path to the checkpoint to load."""
    ## TODO: For now, we only support loading from a local path, since EqV2 checkpoints don't provide public access. In the future, we should support loading from a URL or name. 

    atoms_to_graph: FAIRChemAtomsToGraphSystemConfig
    """Configuration for converting ASE Atoms to a graph."""
    # TODO: Add functionality to load the atoms to graph config from the checkpoint

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the fairchem module is available
        if importlib.util.find_spec("fairchem") is None:
            raise ImportError(
                "The fairchem module is not installed. Please install it by running"
                " pip install fairchem-core."
            )

    @override
    def create_model(self):
        return EqV2BackboneModule(self)


def _combine_scalar_irrep2(stress_head, scalar, irrep2):
    # Change of basis to compute a rank 2 symmetric tensor

    vector = torch.zeros((scalar.shape[0], 3), device=scalar.device).detach()
    flatten_irreps = torch.cat([scalar.reshape(-1, 1), vector, irrep2], dim=1)
    stress = torch.einsum(
        "ab, cb->ca",
        stress_head.block.change_mat.to(flatten_irreps.device),
        flatten_irreps,
    )

    # stress = rearrange(
    #     stress,
    #     "b (three1 three2) -> b three1 three2",
    #     three1=3,
    #     three2=3,
    # ).contiguous()
    stress = stress.view(-1, 3, 3)

    return stress


def _get_pretrained_model(hparams: EqV2BackboneConfig):
    with optional_import_error_message("fairchem"):
        from fairchem.core.common.registry import registry  # type: ignore[reportMissingImports] # noqa
        from fairchem.core.common.utils import update_config  # type: ignore[reportMissingImports] # noqa

    if isinstance(checkpoint_path := hparams.checkpoint_path, CE.CachedPath):
        checkpoint_path = checkpoint_path.resolve()

    checkpoint = None
    # Loads the config from the checkpoint directly (always on CPU).
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config = checkpoint["config"]

    config["trainer"] = config.get("trainer", "ocp")

    if "model_attributes" in config:
        config["model_attributes"]["name"] = config.pop("model")
        config["model"] = config["model_attributes"]

    # Calculate the edge indices on the fly
    config["model"]["otf_graph"] = True

    ### backwards compatability with OCP v<2.0
    config = update_config(config)

    # Save config so obj can be transported over network (pkl)
    config["checkpoint"] = checkpoint_path
    del config["dataset"]["src"]

    # Import a bunch of modules so that the registry can find the classes
    with optional_import_error_message("fairchem"):
        import fairchem.core.models  # type: ignore[reportMissingImports] # noqa
        import fairchem.core.models.equiformer_v2  # type: ignore[reportMissingImports] # noqa
        import fairchem.core.models.equiformer_v2.equiformer_v2  # type: ignore[reportMissingImports] # noqa
        import fairchem.core.models.equiformer_v2.prediction_heads.rank2  # type: ignore[reportMissingImports] # noqa
        import fairchem.core.trainers  # type: ignore[reportMissingImports] # noqa

    trainer_cls = cast(Any, registry.get_trainer_class(config["trainer"]))
    # ^ The typing for FAIRChem's registry is very weird, so we have to do some hacky casting here.
    trainer = trainer_cls(
        task=config.get("task", {}),
        model=config["model"],
        dataset=[config["dataset"]],
        outputs=config["outputs"],
        loss_functions=config["loss_functions"],
        evaluation_metrics=config["evaluation_metrics"],
        optimizer=config["optim"],
        identifier="",
        slurm=config.get("slurm", {}),
        local_rank=config.get("local_rank", 0),
        is_debug=config.get("is_debug", True),
        cpu=True,
        amp=config.get("amp", False),
        inference_only=True,
    )

    # Load the checkpoint
    if checkpoint_path is not None:
        try:
            trainer.load_checkpoint(checkpoint_path, checkpoint, inference_only=True)
        except NotImplementedError:
            log.warning(f"Unable to load checkpoint from {checkpoint_path}")

    # Now, extract the backbone from the trainer and delete the trainer
    with optional_import_error_message("fairchem"):
        from fairchem.core.trainers import OCPTrainer  # type: ignore[reportMissingImports] # noqa

    assert isinstance(trainer, cast(type, OCPTrainer)), "Only OCPTrainer is supported."
    assert (model := getattr(trainer, "_unwrapped_model", None)) is not None, (
        "The model could not be extracted from the trainer. Please report this issue."
    )

    # Make sure this is eqv2
    from fairchem.core.models.base import HydraModel  # type: ignore[reportMissingImports] # noqa

    assert isinstance(model, cast(type, HydraModel)), (
        f"Expected model to be of type HydraModel, but got {type(model)}"
    )
    
    normalizers = trainer.normalizers
    elementrefs = trainer.elementrefs

    return model, normalizers, elementrefs


@final
class EqV2BackboneModule(FinetuneModuleBase["BaseData", "Batch", EqV2BackboneConfig]):
    @override
    @classmethod
    def hparams_cls(cls):
        return EqV2BackboneConfig

    @override
    def requires_disabled_inference_mode(self):
        return False

    def _create_output_head(self, prop: props.PropertyConfig, pretrained_model):
        from fairchem.core.models.base import HydraModel  # type: ignore[reportMissingImports] # noqa

        assert isinstance(pretrained_model, cast(type, HydraModel)), (
            f"Expected model to be of type HydraModel, but got {type(pretrained_model)}"
        )

        match prop:
            case props.EnergyPropertyConfig():
                with optional_import_error_message("fairchem"):
                    from fairchem.core.models.equiformer_v2.equiformer_v2 import (  # type: ignore[reportMissingImports] # noqa
                        EquiformerV2EnergyHead,
                    )
                if self.hparams.reset_output_heads:
                    return EquiformerV2EnergyHead(self.backbone, reduce="sum")
                else:
                    return pretrained_model.output_heads["energy"]

            case props.ForcesPropertyConfig():
                assert not prop.conservative, (
                    "Conservative forces are not supported for eqV2 (yet)"
                )

                with optional_import_error_message("fairchem"):
                    from fairchem.core.models.equiformer_v2.equiformer_v2 import (  # type: ignore[reportMissingImports] # noqa
                        EquiformerV2ForceHead,
                    )
                if self.hparams.reset_output_heads:
                    return EquiformerV2ForceHead(self.backbone)
                else:
                    return pretrained_model.output_heads["forces"]

            case props.StressesPropertyConfig():
                assert not prop.conservative, (
                    "Conservative stresses are not supported for eqV2 (yet)"
                )

                with optional_import_error_message("fairchem"):
                    from fairchem.core.models.equiformer_v2.prediction_heads.rank2 import (  # type: ignore[reportMissingImports] # noqa
                        Rank2SymmetricTensorHead,
                    )

                if self.hparams.reset_output_heads:
                    return Rank2SymmetricTensorHead(
                        self.backbone,
                        output_name="stress",
                        use_source_target_embedding=True,
                        decompose=True,
                        extensive=False,
                    )
                else:
                    return pretrained_model.output_heads["stress"]

            case props.GraphPropertyConfig():
                assert prop.reduction in ("sum", "mean"), (
                    f"Unsupported reduction: {prop.reduction} for eqV2. "
                    "Please use 'sum' or 'mean'."
                )
                if not self.hparams.reset_output_heads:
                    raise ValueError(
                        "Pretrained model does not support general graph properties, only energy, forces, and stresses are supported."
                    )
                return EqV2ScalarHead(self.backbone, reduce=prop.reduction)
            case _:
                raise ValueError(
                    f"Unsupported property config: {prop} for eqV2"
                    "Please ask the maintainers of eqV2 for support"
                )

    @override
    def create_model(self):
        from fairchem.core.models.equiformer_v2.equiformer_v2 import (  # type: ignore[reportMissingImports] # noqa
            EquiformerV2Backbone,
        )

        # Get the pre-trained backbone
        pretrained_model, normalizers, elementrefs = _get_pretrained_model(self.hparams)

        assert isinstance(
            backbone := pretrained_model.backbone, cast(type, EquiformerV2Backbone)
        ), (
            f"Expected backbone to be of type EquiformerV2Backbone, but got {type(backbone)}"
        )

        self.backbone = backbone.float()
        if self.hparams.use_pretrained_normalizers:
            self.pretrained_normalizers = nn.ModuleDict(normalizers)
            self.pretrained_elementrefs = nn.ModuleDict(elementrefs)

        # Create the output heads
        self.output_heads = nn.ModuleDict()
        for prop in self.hparams.properties:
            self.output_heads[prop.name] = self._create_output_head(
                prop, pretrained_model
            ).float()

    @override
    def trainable_parameters(self):
        if not self.hparams.freeze_backbone:
            yield from self.backbone.named_parameters()
        for head in self.output_heads.values():
            yield from head.named_parameters()

    @override
    @contextlib.contextmanager
    def model_forward_context(self, data, mode: str):
        yield

    @override
    def model_forward(self, batch, mode: str, return_backbone_output=False):
        # Run the backbone
        if mode == "predict":
            self.eval()
        emb = self.backbone(batch)

        # Feed the backbone output to the output heads
        predicted_properties: dict[str, torch.Tensor] = {}
        for name, head in self.output_heads.items():
            assert (
                prop := next(
                    (p for p in self.hparams.properties if p.name == name), None
                )
            ) is not None, (
                f"Property {name} not found in properties. "
                "This should not happen, please report this."
            )

            head_output: dict[str, torch.Tensor] = head(batch, emb)

            match prop:
                case props.EnergyPropertyConfig():
                    pred = head_output["energy"].view(-1, 1)
                    if self.hparams.use_pretrained_normalizers:
                        pred = self.pretrained_normalizers["energy"](pred)
                    pred = pred.squeeze(-1)
                case props.ForcesPropertyConfig():
                    pred = head_output["forces"]
                    if self.hparams.use_pretrained_normalizers:
                        pred = self.pretrained_normalizers["forces"](pred)
                case props.StressesPropertyConfig():
                    # Convert the stress tensor to the full 3x3 form
                    stress_rank0 = head_output["stress_isotropic"]  # (bsz 1)
                    if self.hparams.use_pretrained_normalizers:
                        stress_rank0 = self.pretrained_normalizers["stress_isotropic"](
                            stress_rank0
                        )
                    stress_rank2 = head_output["stress_anisotropic"]  # (bsz, 5)
                    if self.hparams.use_pretrained_normalizers:
                        stress_rank2 = self.pretrained_normalizers["stress_anisotropic"](
                            stress_rank2
                        )
                    pred = _combine_scalar_irrep2(head, stress_rank0, stress_rank2)
                case props.GraphPropertyConfig():
                    pred = head_output["energy"]
                case _:
                    assert_never(prop)

            predicted_properties[name] = pred

        pred_dict: ModelOutput = {"predicted_properties": predicted_properties}
        if return_backbone_output:
            pred_dict["backbone_output"] = {"node_embedding": emb["node_embedding"].embedding}
            
        if mode == "predict":
            self.train()

        return pred_dict

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
        with optional_import_error_message("fairchem"):
            from fairchem.core.datasets import data_list_collater  # type: ignore[reportMissingImports] # noqa

        return cast("Batch", data_list_collater(data_list, otf_graph=True))

    @override
    def gpu_batch_transform(self, batch):
        return batch

    @override
    def batch_to_labels(self, batch):
        HARDCODED_NAMES: dict[type[props.PropertyConfigBase], str] = {
            props.EnergyPropertyConfig: "energy",
            props.ForcesPropertyConfig: "forces",
            props.StressesPropertyConfig: "stress",
        }

        labels: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            batch_prop_name = HARDCODED_NAMES.get(type(prop), prop.name)
            labels[prop.name] = batch[batch_prop_name]

        return labels

    @override
    def atoms_to_data(self, atoms, has_labels):
        with optional_import_error_message("fairchem"):
            from fairchem.core.preprocessing import AtomsToGraphs  # type: ignore[reportMissingImports] # noqa

        energy = False
        forces = False
        stress = False
        data_keys = None
        if has_labels:
            energy = any(
                isinstance(prop, props.EnergyPropertyConfig)
                for prop in self.hparams.properties
            )
            forces = any(
                isinstance(prop, props.ForcesPropertyConfig)
                for prop in self.hparams.properties
            )
            stress = any(
                isinstance(prop, props.StressesPropertyConfig)
                for prop in self.hparams.properties
            )
            data_keys = [
                prop.name
                for prop in self.hparams.properties
                if not isinstance(
                    prop,
                    (
                        props.EnergyPropertyConfig,
                        props.ForcesPropertyConfig,
                        props.StressesPropertyConfig,
                    ),
                )
            ]

        a2g = AtomsToGraphs(
            max_neigh=self.hparams.atoms_to_graph.max_num_neighbors,
            radius=cast(
                int, self.hparams.atoms_to_graph.radius
            ),  # Stupid typing of the radius arg by the FAIRChem devs; it should be a float.
            r_energy=energy,
            r_forces=forces,
            r_stress=stress,
            r_data_keys=data_keys,
            r_distances=False,
            r_edges=False,
            r_pbc=True,
        )
        data = a2g.convert(atoms)

        # Reshape the cell and stress tensors to (1, 3, 3)
        #   so that they can be properly batched by the collate_fn.
        if hasattr(data, "cell"):
            data.cell = data.cell.reshape(1, 3, 3)
        if hasattr(data, "stress"):
            data.stress = data.stress.reshape(1, 3, 3).float()
        if hasattr(data, "energy"):
            data.energy = torch.tensor(float(data.energy), dtype=torch.float32)
        if hasattr(data, "forces"):
            data.forces = data.forces.float()
        return data

    @override
    def create_normalization_context_from_batch(self, batch):
        with optional_import_error_message("torch_scatter"):
            from torch_scatter import scatter  # type: ignore[reportMissingImports] # noqa

        atomic_numbers: torch.Tensor = batch["atomic_numbers"].long()  # (n_atoms,)
        batch_idx: torch.Tensor = batch["batch"]  # (n_atoms,)

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
        return NormalizationContext(compositions=compositions)