from backbone import JMPBackbone, JMPBackboneConfig
from jmppeft.utils.goc_graph import Cutoffs, MaxNeighbors
from jmp_data_module import JMPDataModule
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
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import torch
import argparse

# torch.set_float32_matmul_precision('medium')


def main(args_dict: dict):
    ## Load DataModule
    data_module = JMPDataModule(
        batch_size=args_dict["batch_size"],
        xyz_path=args_dict["xyz_path"],
        num_workers=4,
        val_split=0.485,
        test_split=0.485,
        shuffle=True,
        ignore_data_errors=True,
        references={"energy": RidgeReferenceConfig(alpha=0.1)},
    )
    
    ## Build FineTune Model
    backbone = JMPBackboneConfig(
        ckpt_path="/nethome/lkong88/workspace/jmp-peft/checkpoints/jmp-s.pt",
        type="jmp_s",
        freeze = True,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors(main=25, aeaint=20, aint=1000, qint=8),
        qint_tags=[0,1,2],
        edge_dropout=None,
        per_graph_radius_graph=True,
    )
    output_heads = [
        ReferencedEnergyOutputHeadConfig(
            target_name = "energy",
            hidden_dim = 256,
            max_atomic_number = 120,
            activation = "SiLU",
            reduction="sum",
            initialization=RandomReferenceInitializationConfig(),
            loss = loss.MACEHuberEnergyLossConfig(delta=0.01),
            loss_coefficient = 1.0,
        ),
        # DirectForceOutputHeadConfig(
        #     target_name="force",
        #     hidden_dim=512,
        #     activation="SiLU",
        # ),
        GradientForceOutputHeadConfig(
            target_name="force",
            energy_target_name="energy",
            loss = loss.MACEHuberLossConfig(delta=0.01),
            loss_coefficient = 10.0,
        ),
        # GradientStressOutputHeadConfig(
        #     target_name="stress",
        #     energy_target_name="energy",
        #     forces = True,
        #     loss = loss.HuberLossConfig(delta=0.01),
        #     loss_coefficient = 0.1,
        # )
    ]
    metrics_module = MetricsModuleConfig(
        metrics = [
            MetricConfig(
                target_name="energy",
                metric_calculator=MAEMetric(),
                normalize_by_num_atoms=True,
            ),
            MetricConfig(
                target_name="force",
                metric_calculator=MAEMetric(),
                normalize_by_num_atoms=False,
            ),
            # MetricConfig(
            #     target_name="stress",
            #     metric_module=MAEMetric(),
            #     normalize_by_num_atoms=False,
            # ),
        ],
        primary_metric = MetricConfig(
                target_name="force",
                metric_calculator=MAEMetric(),
                normalize_by_num_atoms=False,
            ),
    )
    optimizer = AdamWConfig(
        lr=8e-5,
        weight_decay=0.01,
        amsgrad=False,
        betas=(0.9, 0.95),
    )
    lr_scheduler = StepLRConfig(
        step_size=50,
        gamma=0.95,
    )
    finetune_config = FinetuneModuleBaseConfig(
        project="jmp-finetune-example",
        run_name="water-ef-test",
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
        max_epochs=1000,
        devices = [1,2,3],
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        accelerator='gpu',  
        strategy='ddp', ## reduction of gradient for force gradient?
        precision="bf16-mixed",
        logger=[wandb_logger, csv_logger],
    )
    trainer.fit(finetune_model, datamodule=data_module)
    
    ## bfgs = BatchBFGS([atoms1, atoms2], calc)
    ## bfgs.run(step=100, fmax=0.01)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--xyz_path", type=str, default="./data/water_processed.xyz")
    args_dict = vars(parser.parse_args())
    main(args_dict)
    