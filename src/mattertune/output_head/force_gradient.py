from typing import Literal, Generic
from typing_extensions import override
import contextlib
import torch
import torch.nn as nn
from ..protocol import TBatch
from ..finetune_model import BackBoneBaseOutput
from .base import OutputHeadBaseConfig
from ..modules.loss import LossConfig, L2MAELossConfig
from .utils.force_scaler import ForceScaler
from .utils.tensor_grad import enable_grad


class GradientForceOutputHeadConfig(OutputHeadBaseConfig, Generic[TBatch]):
    """
    Configuration of the GradientForceOutputHead
    Compute force from the gradient of the energy with respect to the position
    """
    ## Paramerters heritated from OutputHeadBaseConfig:
    head_name: str = "GradientForceOutputHead"
    """The name of the output head"""
    target_name: str = "gradient_forces"
    """The name of the target output by this head"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""
    ## New parameters:
    loss: LossConfig = L2MAELossConfig()
    """The loss configuration for the target."""
    energy_target_name: str
    
    @override
    def is_classification(self) -> bool:
        return False
    
    @override
    def construct_output_head(
        self,
        hidden_dim: int|None,
        activation_cls: type[nn.Module],
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
    
    
class GradientForceOutputHead(nn.Module, Generic[TBatch, BackBoneBaseOutput]):
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
        backbone_output: BackBoneBaseOutput,
        output_head_results: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        energy = output_head_results[self.head_config.energy_target_name]
        natoms_in_batch = batch_data.pos.shape[0]
        assert energy.requires_grad, f"Energy tensor {self.head_config.energy_target_name} does not require grad"
        assert batch_data.pos.requires_grad, "Position tensor does not require grad"
        assert energy.shape[0] == batch_data.pos.shape[0], f"Mismatched shapes: energy.shape[0]={energy.shape[0]} mismatch pos.shape[0]={batch_data.pos.shape[0]}. Check your <energy head> and <energy_target_name>"
        forces = self.force_scaler.calc_forces(
            energy,
            batch_data.pos,
        )
        assert forces.shape == (natoms_in_batch, 3), f"forces.shape={forces.shape} != [num_nodes_in_batch, 3]"
        output_head_results[self.head_config.target_name] = forces
        return forces
        