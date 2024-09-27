from pathlib import Path
from typing import Annotated, Literal, TypeAlias, Any
from pydantic import BaseModel, Field
import contextlib
import torch.utils.checkpoint


class PretrainedCheckpointConfig(BaseModel):
    name: Literal["pretrained"] = "pretrained"
    path: Path
    """
    Path to the pretrain checkpoint
    """
    checkpoint_load_args: dict[str, Any] = {}
    """
    Arguments to pass to the checkpoint load function of BackBoneBaseModule
    """


class ResumeCheckpointConfig(BaseModel):
    name: Literal["resume"] = "resume"

    path: Path
    """
    Path to the resume checkpoint
    """


CheckpointConfig: TypeAlias = Annotated[
    PretrainedCheckpointConfig | ResumeCheckpointConfig, Field(discriminator="kind")
]


class CheckpointLoadConfig(BaseModel):
    ignored_key_patterns: list[str] = []
    """Patterns to ignore when loading the checkpoint"""

    ignored_missing_keys: list[str] = []
    """Keys to ignore if they are missing in the checkpoint"""

    ignored_unexpected_keys: list[str] = []
    """Keys to ignore if they are unexpected in the checkpoint"""

    reset_embeddings: bool = False
    """
    If true, it will reset the embeddings to the initial state
    after loading the checkpoint
    """

    checkpoint: CheckpointConfig | None = None
    """
    Configuration for loading the checkpoint
    """
    
    
class GradientCheckpointingConfig(BaseModel):
    preserve_rng_state: bool = False
    """
    Whether to preserve the RNG state when checkpointing.
    Incurs a small overhead if set to `True`.
    """

    use_reentrant: bool = False
    """
    Whether to use reentrant checkpointing.
    This is recommended to be `False`, see https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
    """

    checkpoint_early_stop: bool | None = None
    """
    Non-reentrant checkpoint stops recomputation as soon as all needed intermediate activations have been recomputed.
    Set this to `True` to enable this optimization. Set to `None` to use the default value from PyTorch.

    See https://pytorch.org/docs/stable/checkpoint.html
    """

    @contextlib.contextmanager
    def context(self):
        """
        Context manager to temporarily set the config.
        """
        with contextlib.ExitStack() as stack:
            if (early_stop := self.checkpoint_early_stop) is not None:
                stack.enter_context(
                    torch.utils.checkpoint.set_checkpoint_early_stop(early_stop)
                )
            yield