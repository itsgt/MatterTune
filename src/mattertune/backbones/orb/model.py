from __future__ import annotations

import contextlib
import importlib.util
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, cast

import nshconfig as C
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._functorch import config as functorch_config
from typing_extensions import assert_never, final, override

from ...finetune import properties as props
from ...finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig, ModelOutput
from ...normalization import NormalizationContext
from ...registry import backbone_registry
from ...util import optional_import_error_message
from ..util import voigt_6_to_full_3x3_stress_torch

if TYPE_CHECKING:
    from orb_models.forcefield.base import AtomGraphs  # type: ignore[reportMissingImports] # noqa


log = logging.getLogger(__name__)


class ORBSystemConfig(C.Config):
    """Config controlling how to featurize a system of atoms."""

    radius: float
    """The radius for edge construction."""
    max_num_neighbors: int
    """The maximum number of neighbours each node can send messages to."""

    def _to_orb_system_config(self):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield.atomic_system import SystemConfig  # type: ignore[reportMissingImports] # noqa

        return SystemConfig(
            radius=self.radius,
            max_num_neighbors=self.max_num_neighbors,
        )


@backbone_registry.register
class ORBBackboneConfig(FinetuneModuleBaseConfig):
    name: Literal["orb"] = "orb"
    """The type of the backbone."""

    pretrained_model: str
    """The name of the pretrained model to load."""

    system: ORBSystemConfig = ORBSystemConfig(radius=6.0, max_num_neighbors=120)
    """The system configuration, controlling how to featurize a system of atoms."""

    freeze_backbone: bool = False
    """Whether to freeze the backbone model."""

    @override
    def create_model(self):
        return ORBBackboneModule(self)

    @override
    @classmethod
    def ensure_dependencies(cls):
        # Make sure the orb_models module is available
        if importlib.util.find_spec("orb_models") is None:
            raise ImportError(
                "The orb_models module is not installed. Please install it by running"
                ' pip install "orb_models@git+https://github.com/nimashoghi/orb_models.git"'
            )
            # NOTE: The 0.4.0 version of `orb_models` doesn't actually fully respect
            #   the `device` argument. We have a patch to fix this, and we have
            #   a PR open to fix this upstream. Until that is merged, users
            #   will need to install the patched version of `orb_models` from our fork:
            #   `pip install "orb_models@git+https://github.com/nimashoghi/orb_models.git"`
            #   PR: https://github.com/orbital-materials/orb_models/pull/35
            # FIXME: Remove this note once the PR is merged.

        # Make sure pynanoflann is available
        if importlib.util.find_spec("pynanoflann") is None:
            raise ImportError(
                "The pynanoflann module is not installed. Please install it by running"
                'pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"'
            )


@final
class ORBBackboneModule(
    FinetuneModuleBase["AtomGraphs", "AtomGraphs", ORBBackboneConfig]
):
    @override
    @classmethod
    def hparams_cls(cls):
        return ORBBackboneConfig

    @override
    def requires_disabled_inference_mode(self):
        return False

    def _create_output_head(self, prop: props.PropertyConfig, pretrained_model):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield.forcefield_heads import (
                EnergyHead,
                ForceHead,
                StressHead,
                GraphHead,
            )

        self.include_forces = False
        self.include_stress = False
        match prop:
            case props.EnergyPropertyConfig():
                if not self.hparams.reset_output_heads:
                    return pretrained_model.graph_head
                else:
                    return EnergyHead(
                        latent_dim=256,
                        num_mlp_layers=1,
                        mlp_hidden_dim=256,
                        reference_energy="vasp-shifted",
                    )

            case props.ForcesPropertyConfig():
                self.include_forces = True

                if prop.conservative:
                    return None
                else:
                    if not self.hparams.reset_output_heads:
                        return pretrained_model.node_head
                    else:
                        return ForceHead(
                            latent_dim=256,
                            num_mlp_layers=1,
                            mlp_hidden_dim=256,
                            remove_mean=False,
                            remove_torque_for_nonpbc_systems=False,
                        )

            case props.StressesPropertyConfig():
                self.include_stress = True
                if prop.conservative:
                    return None
                else:
                    if not self.hparams.reset_output_heads:
                        return pretrained_model.stress_head
                    else:
                        return StressHead(
                            latent_dim=256,
                            num_mlp_layers=1,
                            mlp_hidden_dim=256,
                        )

            case props.GraphPropertyConfig():
                with optional_import_error_message("orb_models"):
                    from orb_models.forcefield.property_definitions import (  # type: ignore[reportMissingImports] # noqa
                        PropertyDefinition,
                    )
                if not self.hparams.reset_output_heads:
                    raise ValueError(
                        "Pretrained model does not support general graph properties, only energy, forces, and stresses are supported."
                    )
                else:
                    return GraphHead(
                        latent_dim=256,
                        num_mlp_layers=1,
                        mlp_hidden_dim=256,
                        target=PropertyDefinition(
                            name=prop.name,
                            dim=1,
                            domain="real",
                        ),
                    )
            case props.AtomInvariantVectorPropertyConfig():
                with optional_import_error_message("orb-models"):
                    from orb_models.forcefield import segment_ops
                    from orb_models.forcefield.nn_util import build_mlp
                
                hidden_dim = prop.additional_head_settings['hidden_channels'] if 'hidden_channels' in prop.additional_head_settings else 256
                num_layers = prop.additional_head_settings['num_layers'] if 'num_layers' in prop.additional_head_settings else 1
    
                def head_forward(
                    self, node_features: torch.Tensor, batch: base.AtomGraphs
                ) -> torch.Tensor:
                    """Forward pass (without inverse transformation)."""
                    input = segment_ops.aggregate_nodes(
                        node_features, batch.n_node, reduction=self.node_aggregation
                    )
                    pred = self.mlp(input)
                    print(input)
                    print(input.shape)
                    print(node_features.shape)
                    return pred.squeeze(-1)

                EnergyHead.forward = head_forward
                
                if not self.hparams.reset_output_heads:
                    raise NotImplementedError
                else:
                    head = EnergyHead(
                        latent_dim=256,
                        num_mlp_layers=num_layers,
                        mlp_hidden_dim=hidden_dim,
                    )
                    head.mlp = build_mlp(
                        input_size=256,
                        hidden_layer_sizes=[hidden_dim] * num_layers,
                        output_size=prop.size,
                        activation='silu',
                        dropout=None,
                        checkpoint=None,
                    )
                
                return head
            case _:
                raise ValueError(
                    f"Unsupported property config: {prop} for ORB"
                    "Please ask the maintainers of ORB for support"
                )

    @override
    def create_model(self):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.direct_regressor import DirectForcefieldRegressor
            from orb_models.forcefield.conservative_regressor import ConservativeForcefieldRegressor

        # Get the pre-trained backbone
        # Load the pre-trained model from the ORB package
        if (
            pretrained_model_fn := pretrained.ORB_PRETRAINED_MODELS.get(
                self.hparams.pretrained_model
            )
        ) is None:
            raise ValueError(
                f"Unknown pretrained model: {self.hparams.pretrained_model}"
            )
        # We load on CPU here as we don't have a device yet.
        try:
            pretrained_model = pretrained_model_fn(device="cpu", compile=False) # type: ignore[reportUnknownArgumentType]
        except Exception as e:
            pretrained_model = pretrained_model_fn(device="cpu")
        # This should never be None, but type checker doesn't know that so we need to check.
        assert pretrained_model is not None, "The pretrained model is not available"

        # This should be a `GraphRegressor` object, so we need to extract the backbone.
        assert isinstance(pretrained_model, DirectForcefieldRegressor) or isinstance(pretrained_model, ConservativeForcefieldRegressor), (
            f"Expected a GraphRegressor object, but got {type(pretrained_model)}"
        )
        if isinstance(pretrained_model, DirectForcefieldRegressor):
            self.conservative = False
        else:
            self.conservative = True
        backbone = pretrained_model.model

        # By default, ORB runs the `load_model_for_inference` function on the model,
        #   which sets the model to evaluation mode and freezes the parameters.
        #   We don't want to do that here, so we'll have to revert the changes.
        for param in backbone.parameters():
            param.requires_grad = True

        backbone = backbone.train()
        self.backbone = backbone

        log.info(
            f'Loaded the ORB pre-trained model "{self.hparams.pretrained_model}". The model '
            f"has {sum(p.numel() for p in self.backbone.parameters()):,} parameters."
        )

        # Create the output heads
        self.output_heads = nn.ModuleDict()
        for prop in self.hparams.properties:
            head = self._create_output_head(prop, pretrained_model)
            # assert head is not None, (
            #     f"Find the head for the property {prop.name} is None"
            # )
            self.output_heads[prop.name] = head # type: ignore[reportUnboundType]

    @override
    def trainable_parameters(self):
        if not self.hparams.freeze_backbone:
            yield from self.backbone.named_parameters()
        for head in self.output_heads.values():
            if head is not None:
                yield from head.named_parameters()
                
    @override
    @contextlib.contextmanager
    def model_forward_context(self, data, mode: str):
        with contextlib.ExitStack() as stack:
            if self.conservative:
                stack.enter_context(torch.enable_grad())
                functorch_config.donated_buffer = False

            vectors, stress_displacement, generator = (
                data.compute_differentiable_edge_vectors()
            )
            assert stress_displacement is not None
            assert generator is not None
            data.system_features["stress_displacement"] = stress_displacement
            data.system_features["generator"] = generator
            data.edge_features["vectors"] = vectors
            yield


    @override
    def model_forward(self, batch, mode: str, using_partition: bool = False):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield.forcefield_utils import compute_gradient_forces_and_stress
        
        # Run the backbone
        out = self.backbone(batch)
        node_features = out["node_features"]
        
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
            if head is not None:
                res = head(node_features, batch)
                if isinstance(res, torch.Tensor):
                    predicted_properties[name] = res
                elif isinstance(res, dict):
                    if mode!="predict":
                        predicted_properties[name] = res[name]
                    else:
                        predicted_properties.update(res)
                else:
                    raise ValueError(
                        f"Invalid output from head {head}: {res}"
                    )
            else:
                assert isinstance(prop, props.ForcesPropertyConfig) or isinstance(prop, props.StressesPropertyConfig), (
                    f"Conservative Property {name} is not a force or stress property."
                )
                assert "energy" in predicted_properties, ("Energy property is not found for conservative property prediction. Please put energy property before the conservative property in the config.")
                if name in predicted_properties:
                    pass
                else:
                    forces, stress, _ = compute_gradient_forces_and_stress(
                        energy=predicted_properties["energy"],
                        positions=batch.node_features["positions"],
                        displacement=batch.system_features["stress_displacement"],
                        cell=batch.system_features["cell"],
                        training=self.training,
                        compute_stress=self.include_stress,
                        generator=batch.system_features["generator"],
                    )
                    if self.include_forces:
                        predicted_properties["forces"] = forces
                    if self.include_stress:
                        predicted_properties["stresses"] = stress # type: ignore[reportUnboundType]
        
        if "stresses" in predicted_properties and predicted_properties["stress"].shape[1] == 6: # type: ignore[reportUnboundType]
            # Convert the stress tensor to the full 3x3 form
            predicted_properties["stresses"] = voigt_6_to_full_3x3_stress_torch(
                predicted_properties["stresses"] # type: ignore[reportUnboundType]
            )
            
        pred_dict: ModelOutput = {"predicted_properties": predicted_properties}
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
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield.base import batch_graphs  # type: ignore[reportMissingImports] # noqa

        return batch_graphs(data_list)

    @override
    def gpu_batch_transform(self, batch):
        return batch

    @override
    def batch_to_labels(self, batch):
        # If the labels are not present, throw.
        if not batch.system_targets and not batch.node_targets:
            raise ValueError("No labels found in the batch.")

        labels: dict[str, torch.Tensor] = {}
        for prop in self.hparams.properties:
            match prop_type := prop.property_type():
                case "system":
                    assert batch.system_targets is not None, "System targets are None"
                    labels[prop.name] = batch.system_targets[prop.name]
                case "atom":
                    assert batch.node_targets is not None, "Node targets are None"
                    labels[prop.name] = batch.node_targets[prop.name]
                case _:
                    assert_never(prop_type)

        return labels

    @override
    def atoms_to_data(self, atoms, has_labels):
        with optional_import_error_message("orb_models"):
            from orb_models.forcefield import atomic_system  # type: ignore[reportMissingImports] # noqa

        # This is the dataset transform; we can't use GPU here.
        # NOTE: The 0.4.0 version of `orb_models` doesn't actually fully respect
        #   the `device` argument. We have a patch to fix this, and we have
        #   a PR open to fix this upstream. Until that is merged, users
        #   will need to install the patched version of `orb_models` from our fork:
        #   `pip install "orb_models@git+https://github.com/nimashoghi/orb_models.git"`
        #   PR: https://github.com/orbital-materials/orb_models/pull/35
        atom_graphs = atomic_system.ase_atoms_to_atom_graphs(
            atoms,
            self.hparams.system._to_orb_system_config(),
            device=torch.device("cpu"),
        )
        
        if has_labels:
            if atom_graphs.system_targets is None:
                atom_graphs = atom_graphs._replace(system_targets={})

            # Making the type checker happy
            assert atom_graphs.system_targets is not None

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

                match prop_type := prop.property_type():
                    case "system":
                        atom_graphs.system_targets[prop.name] = (
                            value.reshape(1, 1) if value.dim() == 0 else value
                        )
                    case "atom":
                        atom_graphs.node_targets[prop.name] = value # type: ignore[reportUnboundType]
                    case _:
                        assert_never(prop_type)

        # For normalization purposes, we should just pre-compute the composition
        #   vector here and save it in the `system_features`. Then, when batching happens,
        #   we can just use that composition vector from the batched `system_features`.
        atom_types_onehot = F.one_hot(
            atom_graphs.atomic_numbers.view(-1).long(),
            num_classes=120,
        )
        # ^ (n_atoms, 120)
        # Now we need to sum this up to get the composition vector
        composition = atom_types_onehot.sum(dim=0, keepdim=True)
        # ^ (1, 120)
        atom_graphs.system_features["norm_composition"] = composition

        return atom_graphs

    @override
    def create_normalization_context_from_batch(self, batch):
        num_atoms = batch.n_node
        compositions = batch.system_features.get("norm_composition")
        if compositions is None:
            raise ValueError("No composition found in the batch.")
        compositions = compositions[:, 1:]  # Remove the zeroth element
        return NormalizationContext(num_atoms=num_atoms, compositions=compositions)
