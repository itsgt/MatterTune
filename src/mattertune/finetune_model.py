import torch
import torch.nn as nn


class FineTuneModel(nn.Module):
    def __init__(self):
        self.backbone = None
        self.output_head = None
        
    def forward(self, x):
        pass