from __future__ import annotations

import argparse

from bandgap_data_module import MatglDataModule
from m3gnet_backbone import M3GNetBackboneConfig
from matbench.bench import MatbenchBenchmark
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from pymatgen.core.structure import Structure
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger

import mattertune.finetune.loss as loss
from mattertune.finetune import (
    FinetuneModuleBase,
    FinetuneModuleBaseConfig,
)
from mattertune.finetune.lr_scheduler import StepLRConfig
from mattertune.finetune.metrics import MAEMetric, MetricConfig, MetricsModuleConfig
from mattertune.finetune.optimizer import AdamWConfig
from mattertune.output_heads.goc_style.heads.global_direct import (
    GlobalScalerOutputHeadConfig,
)


def main(args_dict: dict):
    mb = MatbenchBenchmark(autoload=False, subset=["matbench_mp_gap"])

    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            # Inputs are either chemical compositions as strings
            # or crystal structures as pymatgen.Structure objects.
            # Outputs are either floats (regression tasks) or bools (classification tasks)
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            structures: list[Structure] = train_inputs
            labels = {
                "band_gap": train_outputs,
            }
            ## Build DataModule
            element_types = get_element_list(structures)
            converter = Structure2Graph(
                element_types=element_types, cutoff=args_dict["cutoff"]
            )
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
                shuffle=False,
                ignore_data_errors=True,
            )

            ## Build FineTune Model
            backbone = M3GNetBackboneConfig(
                path="M3GNet-MP-2021.2.8-PES",
                freeze=args_dict["freeze_backbone"],
            )
            output_heads = [
                GlobalScalerOutputHeadConfig(
                    target_name="band_gap",
                    hidden_dim=64,
                    activation="SiLU",
                    loss=loss.MACEHuberEnergyLossConfig(delta=0.01),
                    reduction="mean",
                )
            ]
            metrics_module = MetricsModuleConfig(
                metrics=[
                    MetricConfig(
                        target_name="band_gap",
                        metric_calculator=MAEMetric(),
                        normalize_by_num_atoms=False,
                    ),
                ],
                primary_metric=MetricConfig(
                    target_name="band_gap",
                    metric_calculator=MAEMetric(),
                    normalize_by_num_atoms=False,
                ),
            )
            optimizer = AdamWConfig(
                lr=args_dict["lr"],
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
            csv_logger = CSVLogger(save_dir="./lightning_logs/", name="csv_logs")
            wandb_logger = WandbLogger(
                project=finetune_config.project, name=finetune_config.run_name
            )
            trainer = Trainer(
                max_epochs=args_dict["max_epochs"],
                devices=args_dict["devices"],
                gradient_clip_algorithm="value",
                gradient_clip_val=1.0,
                accelerator="gpu",
                strategy="ddp_find_unused_parameters_true",
                precision="bf16-mixed",
                logger=[wandb_logger, csv_logger],
            )
            trainer.fit(finetune_model, datamodule=data_module)

            # Get testing data
            test_inputs = task.get_test_data(fold, include_target=False)
            predict_dataloader = data_module.get_predict_dataloader(test_inputs)
            print("Number of test samples: ", len(test_inputs))
            # Predict on the testing data
            # Your output should be a pandas series, numpy array, or python iterable
            # where the array elements are floats or bools
            prediction = finetune_model.predict_distributed(predict_dataloader)
            prediction = prediction["band_gap"].detach().cpu().numpy()
            # print(prediction.shape)
            prediction = prediction.reshape(-1).tolist()

            # Record your data!
            task.record(fold, prediction)

            # Close the logger
            wandb_logger._experiment.finish()
    mb.to_file("matgl-tune-mp-bandgap.json.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--threebody_cutoff", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.0)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 3])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
