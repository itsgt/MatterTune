from abc import abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl
from mattertune.protocol import (
    TData,
    TBatch,
    BackBoneBaseOutput,
    BackBoneBaseModule,
    OutputHeadBaseConfig,
)

class FineTunebaseModule(pl.LightningModule):
    """
    The base class of Finetune Model heritates from pytorch_lightning.LightningModule
    Two main components:
    - backbone: BackBoneBaseModel loaded from the pretrained model
    - output_head: defined by the user in finetune task
    """
    def __init__(
        self, 
        backbone: BackBoneBaseModule,
        output_heads_config: list[OutputHeadBaseConfig],
        **kwargs,
    ):
        super(FineTunebaseModule, self).__init__()
        self.backbone = backbone
        self.output_heads_config = output_heads_config
    
    