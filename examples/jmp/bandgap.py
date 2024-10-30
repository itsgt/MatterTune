from matbench.bench import MatbenchBenchmark
from jmp_backbone import JMPBackboneConfig
from jmppeft.utils.goc_graph import Cutoffs, MaxNeighbors
from mattertune.data_structures import RawASEAtomsDataProviderConfig
from mattertune.output_heads.goc_style.heads import GlobalScalerOutputHeadConfig
import mattertune.finetune.loss as loss
from mattertune.finetune.metrics import MetricConfig, MAEMetric, MetricsModuleConfig
from mattertune.finetune.optimizer import AdamWConfig
from mattertune.finetune.lr_scheduler import StepLRConfig
from mattertune.finetune.base import FinetuneModuleBase, FinetuneModuleBaseConfig
from mattertune.potential import MatterTunePotential
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger


def main(args_dict:dict):
    mb = MatbenchBenchmark(autoload=False, subset=["matbench_mp_gap"])
    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            atoms_list = [AseAtomsAdaptor.get_atoms(struct) for struct in train_inputs]
            labels = [{"band_gap": torch.tensor([bg], dtype=torch.float)} for bg in train_outputs]
            ## Set DataProvider Config
            data_provider = RawASEAtomsDataProviderConfig(
                atoms_list=atoms_list,
                labels=labels,
                val_split=args_dict["val_split"],
                test_split=0.0,
                shuffle=False,
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
                GlobalScalerOutputHeadConfig(
                    target_name = "band_gap",
                    hidden_dim = 256,
                    activation="SiLU",
                    loss = loss.MACEHuberEnergyLossConfig(delta=0.01),
                    reduction="mean",
                )
            ]
            ## Set MetricsModule Config
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
            ## Set Optimizer Config
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
            
            ## Setup FineTuneModule
            finetune_config = FinetuneModuleBaseConfig(
                project="MatterTune-Example",
                run_name="jmp-matbench-bandgap",
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
            simplified_config = {
                "project": finetune_config.project,
                "run_name": finetune_config.run_name,
                "max_epochs": args_dict["max_epochs"],
                "batch_size": args_dict["batch_size"],
                "val_split": args_dict["val_split"],
                # Add other simple scalar configurations as needed
            }
            wandb_logger = WandbLogger(project=finetune_config.project, name=finetune_config.run_name, config=simplified_config)
            trainer = Trainer(
                max_epochs=args_dict["max_epochs"],
                devices = args_dict["gpus"],
                gradient_clip_algorithm="value",
                gradient_clip_val=1.0,
                accelerator='gpu',  
                strategy='ddp',
                precision="bf16-mixed",
                logger=[wandb_logger],
                default_root_dir="./checkpoints/bandgap",
            )
            trainer.fit(finetune_model)
            ## Load Best Checkpoint
            # Find the checkpoint callback and get the best model path
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
            else:
                raise ValueError("No checkpoint callback found in the trainer.")
            
            test_inputs = task.get_test_data(fold, include_target=False)
            atoms_list = [AseAtomsAdaptor.get_atoms(struct) for struct in test_inputs]
            potential = MatterTunePotential(model = model, trainer = trainer, batch_size=args_dict["batch_size"], print_log=True)
            predictions_dict: dict[str, torch.Tensor] = potential.predict(atoms_list)
            predictions = predictions_dict["band_gap"].detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
            task.record(fold, predictions)
            print(f"Fold {fold} completed")
    mb.to_file("matbench_mp_gap.json.gz")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0,1,2])
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()
    
    main(vars(args))