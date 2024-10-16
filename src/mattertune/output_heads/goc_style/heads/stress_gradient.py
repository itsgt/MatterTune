from typing import Literal, Generic
from typing_extensions import override
import contextlib
import torch
import torch.nn as nn
from einops import rearrange
from mattertune.protocol import TBatch
from mattertune.output_heads.base import OutputHeadBaseConfig
from mattertune.finetune.loss import LossConfig, MAELossConfig
from mattertune.output_heads.goc_style.heads.utils.force_scaler import ForceStressScaler
from mattertune.output_heads.goc_style.heads.utils.tensor_grad import enable_grad
from mattertune.output_heads.goc_style.backbone_module import GOCStyleBackBoneOutput


class GradientStressOutputHeadConfig(OutputHeadBaseConfig, Generic[TBatch]):
    r"""
    Description of this layer:

    **Before the backbone forward pass:**

    1. **Create the displacement tensor**: We begin by creating a small displacement tensor, which represents an infinitesimal deformation of the system. This tensor is denoted as $\mathbf{displacement}$.

    2. **Compute the symmetric part of the displacement tensor**: We then compute the symmetric part of the displacement tensor, denoted as $\mathbf{symmetric\_displacement}$, using the formula:

    $\mathbf{symmetric\_displacement} = \frac{1}{2}\left(\mathbf{displacement} + \mathbf{displacement}^\top\right)$

    This ensures that the displacement tensor is symmetric, as the stress tensor should be.

    3. **Apply the deformation to the atom positions**: We apply the symmetric displacement tensor to the atom positions using the following formula:

    $\mathbf{r}' = \mathbf{r} + \mathbf{r} \cdot \mathbf{symmetric\_displacement}$

    Here, $\mathbf{r}$ and $\mathbf{r}'$ are the original and deformed atom positions, respectively.

    4. **Apply the deformation to the cell (if present)**: If the cell information is available, we apply the symmetric displacement tensor to the cell vectors using the following formula:

    $\mathbf{h}' = \mathbf{h} + \mathbf{h} \cdot \mathbf{symmetric\_displacement}$

    Here, $\mathbf{h}$ and $\mathbf{h}'$ are the original and deformed cell vectors, respectively.

    **After the backbone forward pass:**

    1. **Compute the virial**: We compute the virial $\mathbf{W}$ as the negative of the gradient of the total energy $E$ with respect to the displacement tensor $\mathbf{displacement}$:

    $\mathbf{W} = -\frac{\partial E}{\partial \mathbf{displacement}}$

    2. **Compute the volume**: If the cell information is available, we compute the volume $V$ as the absolute value of the determinant of the cell vectors $\mathbf{h}$:

    $V = |\det(\mathbf{h})|$

    3. **Compute the stress tensor**: The stress tensor $\mathbf{\sigma}$ is defined as the negative of the virial $\mathbf{W}$ divided by the volume $V$:

    $\mathbf{\sigma} = -\frac{1}{V}\mathbf{W}$

    The key equations used in this process are:

    $\mathbf{symmetric\_displacement} = \frac{1}{2}\left(\mathbf{displacement} + \mathbf{displacement}^\top\right)$
    $\mathbf{r}' = \mathbf{r} + \mathbf{r} \cdot \mathbf{symmetric\_displacement}$
    $\mathbf{h}' = \mathbf{h} + \mathbf{h} \cdot \mathbf{symmetric\_displacement}$
    $\mathbf{W} = -\frac{\partial E}{\partial \mathbf{displacement}}$
    $V = |\det(\mathbf{h})|$
    $\mathbf{\sigma} = -\frac{1}{V}\mathbf{W}$
    """
    
    ## Paramerters heritated from OutputHeadBaseConfig:
    pred_type: Literal["scalar", "vector", "tensor", "classification"] = "tensor"
    """The prediction type of the output head"""
    target_name: str = "gradient_stress"
    """The name of the target output by this head"""
    ## New parameters:
    loss: LossConfig = MAELossConfig()
    """The loss function to use for the target"""
    forces: bool = False
    """Whether to compute the forces as well"""
    energy_target_name: str

    @override
    def is_classification(self) -> bool:
        return False

    @override
    def construct_output_head(
        self,
    ):
        return GradientStressOutputHead(self)

    @override
    @contextlib.contextmanager
    def model_forward_context(self, data: TBatch):
        if not hasattr(data, "cell_displacement"):
            raise ValueError(
                "cell_displacement must be provided for GradientStressOutputHead, Can be all zeros"
            )
        
        with contextlib.ExitStack() as stack:
            enable_grad(stack)

            if not data.pos.requires_grad:
                data.pos.requires_grad_(True)

            # 初始化 displacement 属性
            num_graphs = int(torch.max(data.batch).item() + 1)
            data.cell_displacement = torch.zeros(
                (num_graphs, 3, 3), dtype=data.pos.dtype, device=data.pos.device
            )
            data.cell_displacement.requires_grad_(True)

            symmetric_displacement = 0.5 * (
                data.cell_displacement + data.cell_displacement.transpose(-1, -2)
            )

            # 初始化 cell 属性
            if data.cell is None:
                data.cell = torch.zeros(
                    (3, 3), dtype=data.pos.dtype, device=data.pos.device
                )
                data.cell.requires_grad_(True)

            data.pos = data.pos + torch.bmm(
                data.pos.unsqueeze(-2), symmetric_displacement[data.batch]
            ).squeeze(-2)
            data.cell = data.cell + torch.bmm(data.cell, symmetric_displacement)

            yield
        
    @override
    def supports_inference_mode(self):
        return False
    

class GradientStressOutputHead(nn.Module, Generic[TBatch]):
    """
    The output head of the gradient stress target.
    """

    @override
    def __init__(
        self, 
        head_config: GradientStressOutputHeadConfig,
    ):
        super().__init__()

        self.head_config = head_config
        if head_config.forces:
            self.force_stress_scaler = ForceStressScaler()

    @override
    def forward(
        self, 
        *,
        batch_data: TBatch,
        backbone_output: GOCStyleBackBoneOutput,
        output_head_results: dict[str, torch.Tensor],
    ) -> torch.Tensor:

        # Displacement must be in data
        cell_displacement = batch_data.cell_displacement
        cell = batch_data.cell
        energy = output_head_results[self.head_config.energy_target_name]
        if cell_displacement is None:
            raise ValueError(
                "cell_displacement must be provided for GradientStressOutputHead, but found None"
            )
        if cell is None:
            raise ValueError("cell must be provided for GradientStressOutputHead, but found None")

        if self.head_config.forces:
            forces, stress = self.force_stress_scaler.calc_forces_and_update(
                energy, batch_data.pos, cell_displacement, cell
            )

            # Store the forces in the input dict so that they can be used
            # by the force head.
            output_head_results["_stress_precomputed_forces"] = forces
        else:
            grad = torch.autograd.grad(
                energy,
                [cell_displacement],
                grad_outputs=torch.ones_like(energy),
                create_graph=self.training,
            )
            # forces = -1 * grad[0]
            virial = grad[0]

            volume = torch.linalg.det(batch_data.cell).abs()
            # tc.tassert(tc.Float[torch.Tensor, "bsz"], volume)
            num_graphs = int(torch.max(batch_data.batch).item() + 1)
            assert volume.shape == (num_graphs,), f"volume.shape={volume.shape} != {(num_graphs,)}"
            assert torch.is_floating_point(volume), f"volume.dtype={volume.dtype}, expected floating point"
            stress = virial / rearrange(volume, "b -> b 1 1")

        return stress