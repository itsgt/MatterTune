import pytorch_lightning as pl
from pytorch_lightning import Trainer
from ase.atoms import Atoms
import torch
from mattertune.finetune.base import FinetuneModuleBase
from mattertune.data_structures import MatterTuneDataSetBase
from torch.utils.data import DataLoader, DistributedSampler
from collections import defaultdict
import torch.distributed as dist
from typing import Literal
import logging

class MatterTunePotential():
    """
    Wrap FinetuneModuleBase as a Potential class
    Provide a unified interface for doing inference
    """
    def __init__(
        self,
        *,
        model: FinetuneModuleBase,
        trainer: Trainer|None = None,
        accelator: Literal["cpu", "gpu"] = "gpu",
        devices: list[int] = [0],
        batch_size: int,
        pin_memory: bool = True,
        num_workers: int = 4,
        print_log: bool = False,
    ):
        self.model = model
        self.model.disable_callbacks()
        self.accelator = accelator
        if trainer is not None:
            self.trainer = trainer
        else:
            if accelator == "cpu":
                self.trainer = Trainer(
                    accelerator=accelator,
                    precision=None,
                    inference_mode=model.inference_mode,
                    enable_model_summary=print_log,
                    enable_progress_bar=print_log,
                    logger=False,
                )
            elif accelator == "gpu":
                self.trainer = Trainer(
                    accelerator=accelator,
                    devices=devices,
                    strategy="ddp",
                    precision="bf16-mixed",
                    inference_mode=model.inference_mode,
                    enable_model_summary=print_log,
                    enable_progress_bar=print_log,
                    logger=False,
                )
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        
    def get_supported_properties(self) -> list[str]:
        supported_properties = []
        output_head_configs = self.model.config.output_heads
        for config in output_head_configs:
            supported_properties.append(config.target_name)
        return supported_properties

    def predict(self, atoms_list: list[Atoms]) -> dict[str, torch.Tensor]:
        """
        Predict the properties of atoms list
        """
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        
        data_list = [self.model.backbone.process_raw(atoms=atoms, idx=i, labels={}, inference=True) for i, atoms in enumerate(atoms_list)]
        dataset = MatterTuneDataSetBase(data_list)
        
        # Using distributed sampler for multi-device scenarios
        multi_device: bool = (self.accelator == "gpu") and (self.trainer.num_devices > 1)
        sampler = DistributedSampler(dataset) if multi_device else None
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=self.model.backbone.collate_fn,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

        # Running prediction using trainer
        predictions = self.trainer.predict(self.model, dataloaders=dataloader)

        # Handling empty predictions case
        if not predictions:
            raise ValueError("No predictions were generated. Please check the model and input data.")

        # Aggregate predictions across devices (e.g., multiple GPUs)
        if multi_device and dist.is_initialized():
            all_predictions = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_predictions, predictions)
            predictions = [pred for device_preds in all_predictions for pred in device_preds]
        
        # Use idx to restore original order, filter invalid entries
        unique_predictions = defaultdict(dict)
        valid_range = set(range(len(atoms_list)))

        for batch in predictions:
            valid_mask = [idx in valid_range for idx in batch["idx"].tolist()]  # Filter out invalid entries
            for key, value in batch.items():
                if key == "idx":
                    for idx, valid in zip(value.tolist(), valid_mask):
                        if valid:
                            unique_predictions[idx]["idx"] = idx
                else:
                    for idx, val, valid in zip(batch["idx"].tolist(), value, valid_mask):
                        if valid:
                            # If the key is not in the dictionary or the key is not in the dictionary with the same index
                            if idx not in unique_predictions or key not in unique_predictions[idx]:
                                unique_predictions[idx][key] = val

        # sort the predictions by idx
        sorted_indices = sorted(unique_predictions.keys())
        concated_predictions = {key: [] for key in unique_predictions[sorted_indices[0]].keys()}

        # Concatenate the predictions
        for idx in sorted_indices:
            for key, value in unique_predictions[idx].items():
                concated_predictions[key].append(value)
        for key in concated_predictions.keys():
            if key != "idx":
                concated_predictions[key] = torch.stack(concated_predictions[key], dim=0)

        return concated_predictions
