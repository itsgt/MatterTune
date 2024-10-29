from __future__ import annotations

import contextlib
from typing import Generic, Literal

import torch
import torch.nn as nn
from typing_extensions import override

from mattertune.finetune.loss import L2MAELossConfig, LossConfig
from mattertune.output_heads.base import OutputHeadBaseConfig
from mattertune.output_heads.goc_style.backbone_module import GOCStyleBackBoneOutput
from mattertune.output_heads.goc_style.heads.utils.force_scaler import ForceScaler
from mattertune.output_heads.goc_style.heads.utils.tensor_grad import enable_grad
from mattertune.protocol import TBatch


class GradientForceOutputHeadConfig(OutputHeadBaseConfig, Generic[TBatch]):
    """
    Configuration of the GradientForceOutputHead
    Compute force from the gradient of the energy with respect to the position
    """

    ## Paramerters heritated from OutputHeadBaseConfig:
    pred_type: Literal["scalar", "vector", "tensor", "classification"] = "vector"
    """The prediction type of the output head"""
    target_name: str = "gradient_forces"
    """The name of the target output by this head"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    ## New parameters:
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""
    loss: LossConfig = L2MAELossConfig()
    """The loss configuration for the target."""
    energy_target_name: str

    @override
    def is_classification(self) -> bool:
        return False

    @override
    def construct_output_head(
        self,
    ) -> nn.Module:
        return GradientForceOutputHead(self)

    @override
    @contextlib.contextmanager
    def model_forward_context(self, data: TBatch):
        with contextlib.ExitStack() as stack:
            enable_grad(stack)

            if not data.pos.requires_grad:
                data.pos.requires_grad_(True)
            yield

    @override
    def supports_inference_mode(self):
        return False


class GradientForceOutputHead(nn.Module, Generic[TBatch]):
    """
    Compute force from the gradient of the energy with respect to the position
    """

    @override
    def __init__(self, head_config: GradientForceOutputHeadConfig):
        super(GradientForceOutputHead, self).__init__()
        self.head_config = head_config
        self.force_scaler = ForceScaler()

    @override
    def forward(
        self,
        *,
        batch_data: TBatch,
        backbone_output: GOCStyleBackBoneOutput,
        output_head_results: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        energy = output_head_results[self.head_config.energy_target_name]
        natoms_in_batch = batch_data.pos.shape[0]
        assert (
            energy.requires_grad
        ), f"Energy tensor {self.head_config.energy_target_name} does not require grad"
        assert batch_data.pos.requires_grad, "Position tensor does not require grad"
        forces = self.force_scaler.calc_forces(
            energy,
            batch_data.pos,
        )
        assert forces.shape == (
            natoms_in_batch,
            3,
        ), f"forces.shape={forces.shape} != [num_nodes_in_batch, 3]"
        output_head_results[self.head_config.target_name] = forces
        return forces
