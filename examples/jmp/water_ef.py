from jmp_backbone import JMPBackboneConfig
from jmppeft.utils.goc_graph import Cutoffs, MaxNeighbors
from mattertune.data_structures import RawEFSDataProviderFromXYZConfig
from mattertune.output_heads.goc_style.heads import (
    ReferencedEnergyOutputHeadConfig,
    RandomReferenceInitializationConfig,
    GradientForceOutputHeadConfig
)
import mattertune.finetune.loss as loss
from mattertune.finetune.metrics import MetricConfig, MAEMetric, MetricsModuleConfig
from mattertune.finetune.optimizer import AdamWConfig
from mattertune.finetune.lr_scheduler import StepLRConfig
from mattertune.finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger


def main(args_dict: dict):
    ## Set DataProvider Config
    data_provider = RawEFSDataProviderFromXYZConfig(
        file_path=args_dict["xyz_path"],
        val_split=(1-args_dict["train_split"])/2,
        test_split=(1-args_dict["train_split"])/2,
        shuffle=True,
        include_forces=True,
        include_stress=False,
    )
    
    ## Set Backbone Config
    backbone = JMPBackboneConfig(
        ckpt_path="/net/csefiles/coc-fung-cluster/lingyu/checkpoints/jmp-s.pt",
        type="jmp_s",
        freeze = True,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors(main=25, aeaint=20, aint=1000, qint=8),
        qint_tags=[0,1,2],
        edge_dropout=None,
        per_graph_radius_graph=True,
    )
    ## Set OutputHeads Config
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
        GradientForceOutputHeadConfig(
            target_name="forces",
            energy_target_name="energy",
            loss = loss.MACEHuberLossConfig(delta=0.01),
            loss_coefficient = 10.0,
        ),
    ]
    ## Set MetricsModule Config
    metrics_module = MetricsModuleConfig(
        metrics = [
            MetricConfig(
                target_name="energy",
                metric_calculator=MAEMetric(),
                normalize_by_num_atoms=True,
            ),
            MetricConfig(
                target_name="forces",
                metric_calculator=MAEMetric(),
                normalize_by_num_atoms=False,
            ),
        ],
        primary_metric = MetricConfig(
                target_name="forces",
                metric_calculator=MAEMetric(),
                normalize_by_num_atoms=False,
            ),
    )
    ## Set Optimizer Config
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
    
    ## Setup FineTuneModule
    finetune_config = FinetuneModuleBaseConfig(
        project="MatterTune-Example",
        run_name="jmp-water-ef",
        backbone=backbone,
        output_heads=output_heads,
        raw_data_provider=data_provider,
        metrics_module=metrics_module,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        batch_size=args_dict["batch_size"],
        num_workers=4,
        ignore_data_errors = False,
        early_stopping_patience=200,
    )
    finetune_model = FinetuneModuleBase(finetune_config)
    
    ## Fit the model
    wandb_logger = WandbLogger(project=finetune_config.project, name=finetune_config.run_name)
    trainer = Trainer(
        max_epochs=args_dict["max_epochs"],
        devices = args_dict["gpus"],
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
        accelerator='gpu',  
        strategy='ddp', ## reduction of gradient for force gradient?
        precision="bf16-mixed",
        logger=[wandb_logger],
        default_root_dir="./checkpoints/water_ef",
        inference_mode=finetune_model.inference_mode,
    )
    trainer.fit(finetune_model)
    
    from pytorch_lightning.callbacks import ModelCheckpoint
            
    best_checkpoint_callback = None
    for callback in trainer.checkpoint_callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_checkpoint_callback = callback
            break

    if best_checkpoint_callback:
        best_checkpoint_path = best_checkpoint_callback.best_model_path
        # Load the best checkpoint
        model = FinetuneModuleBase.load_from_checkpoint(best_checkpoint_path)
        model.raw_data_provider = data_provider
    else:
        raise ValueError("No checkpoint callback found in the trainer.")
    
    trainer.test(model)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--xyz_path", type=str, default="../data/water_processed.xyz")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0,3])
    parser.add_argument("--train_split", type=float, default=0.03)
    args_dict = vars(parser.parse_args())
    main(args_dict)