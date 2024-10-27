from m3gnet_backbone import M3GNetBackbone, M3GNetBackboneConfig
from bandgap_data_module import MatglDataModule
from mp_api.client import MPRester
import numpy as np
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from mattertune.finetune import (
    FinetuneModuleBaseConfig, 
    FinetuneModuleBase,
)
from mattertune.finetune.data_module import RidgeReferenceConfig
from mattertune.finetune.metrics import MetricConfig, MAEMetric, MetricsModuleConfig
from mattertune.finetune.optimizer import AdamConfig, AdamWConfig
from mattertune.finetune.lr_scheduler import StepLRConfig, CosineAnnealingLRConfig
import mattertune.finetune.loss as loss
from mattertune.output_heads.goc_style.heads.scaler_referenced import (
    ReferencedEnergyOutputHeadConfig,
    RandomReferenceInitializationConfig,
    ZerosReferenceInitializationConfig,
)
from mattertune.output_heads.goc_style.heads.force_gradient import GradientForceOutputHeadConfig
from mattertune.output_heads.goc_style.heads.stress_gradient import GradientStressOutputHeadConfig
from mattertune.output_heads.goc_style.heads.force_direct import DirectForceOutputHeadConfig
from mattertune.output_heads.goc_style.heads.stress_direct import DirectStressOutputHeadConfig
from mattertune.output_heads.goc_style.heads.global_direct import GlobalScalerOutputHeadConfig
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import torch
import argparse


def main(args_dict:dict):
    # TODO: use matbench
    
    api_key = "m6MxOKC4V7487mjSRrhHYFayktrzUCJW"
    with MPRester(api_key) as mpr:
        search_results = mpr.summary.search(
            fields=["material_id", "formula_pretty", "band_gap",],
        )
        materials_data = list(search_results)
        structures = []
        band_gaps = []
        for i, material in enumerate(materials_data):
            material_id = material.material_id
            band_gap = material.band_gap
            structure = mpr.get_structure_by_material_id(material_id)
            if structure is not None and band_gap is not None and band_gap > 0.0:
                structures.append(structure)
                band_gaps.append(band_gap)
        print(f"{len(structures)} structures with band gaps > 0.0 loaded from MP.")
        labels = {
            "band_gap": band_gaps,
        }
        
    ## Build DataModule
    element_types = get_element_list(structures)
    converter = Structure2Graph(element_types=element_types, cutoff=args_dict["cutoff"])
    data_module = MatglDataModule(
        batch_size=args_dict["batch_size"],
        structures=structures,
        labels=labels,
        threebody_cutoff=args_dict["threebody_cutoff"],
        converter=converter,
        include_line_graph=False,
        directed_line_graph=False,
        num_workers=4,
        val_split=args_dict["val_split"],
        test_split=args_dict["test_split"],
        shuffle=True,
        ignore_data_errors=True,
    )
    
    ## Build FineTune Model
    backbone = M3GNetBackboneConfig(
        path="M3GNet-MP-2021.2.8-PES",
        freeze=args_dict["freeze_backbone"],
    )
    output_heads = [
        GlobalScalerOutputHeadConfig(
            target_name = "band_gap",
            hidden_dim = 64,
            activation="SiLU",
            loss = loss.MACEHuberEnergyLossConfig(delta=0.01),
            reduction="mean",
        )
    ]
    metrics_module = MetricsModuleConfig(
        metrics = [
            MetricConfig(
                target_name="band_gap",
                metric_calculator=MAEMetric(),
                normalize_by_num_atoms=False,
            ),
        ],
        primary_metric = MetricConfig(
                target_name="band_gap",
                metric_calculator=MAEMetric(),
                normalize_by_num_atoms=False,
            ),
    )
    optimizer = AdamWConfig(
        lr=1e-3,
        weight_decay=0.01,
        amsgrad=False,
        betas=(0.9, 0.95),
    )
    lr_scheduler = StepLRConfig(
        step_size=50,
        gamma=0.95,
    )
    finetune_config = FinetuneModuleBaseConfig(
        project="matgl-finetune-example",
        run_name="mp_band_gap",
        backbone=backbone,
        output_heads=output_heads,
        metrics_module=metrics_module,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        early_stopping_patience=None,
    )
    finetune_model = FinetuneModuleBase(finetune_config)
    
    ## Train Model
    csv_logger = CSVLogger(save_dir='./lightning_logs/', name='csv_logs')
    wandb_logger = WandbLogger(project=finetune_config.project, name=finetune_config.run_name)
    trainer = Trainer(
        max_epochs=args_dict["max_epochs"],
        devices = args_dict["devices"],
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        accelerator='gpu',  
        strategy='ddp',
        precision="bf16-mixed",
        logger=[wandb_logger, csv_logger],
    )
    trainer.fit(finetune_model, datamodule=data_module)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--threebody_cutoff", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)