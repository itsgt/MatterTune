from __future__ import annotations

import logging
from pathlib import Path

import mattertune.configs as MC
import nshutils as nu
from mattertune import MatterTuner
from mattertune.configs import WandbLoggerConfig

logging.basicConfig(level=logging.DEBUG)
nu.pretty()


def main(args_dict: dict):
    def hparams():
        hparams = MC.MatterTunerConfig.draft()

        ## Model Hyperparameters
        hparams.model = MC.M3GNetBackboneConfig.draft()
        hparams.model.graph_computer = MC.M3GNetGraphComputerConfig.draft()
        hparams.model.graph_computer.pre_compute_line_graph = True
        hparams.model.ckpt_path = Path(args_dict["ckpt_path"])
        hparams.model.ignore_gpu_batch_transform_error = True
        hparams.model.optimizer = MC.AdamWConfig(lr=args_dict["lr"])
        hparams.model.lr_scheduler = MC.CosineAnnealingLRConfig(
            T_max=args_dict["max_epochs"]
        )

        # Add model properties
        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(
            loss=MC.MAELossConfig(),
            loss_coefficient=0.01,
        )
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MAELossConfig(),
            conservative=True,
            loss_coefficient=50.0
        )
        hparams.model.properties.append(forces)
        stress = MC.StressesPropertyConfig(
            loss=MC.MAELossConfig(),
            loss_coefficient=50.0,
            conservative=True
        )
        hparams.model.properties.append(stress)

        ## Data Hyperparameters
        hparams.data = MC.AutoSplitDataModuleConfig.draft()
        hparams.data.dataset = MC.JSONDatasetConfig.draft()
        tasks = {
            "energy": args_dict["energy_attr"],
            "forces": args_dict["forces_attr"],
            "stress": args_dict["stress_attr"],
        }
        hparams.data.dataset.tasks = tasks
        hparams.data.dataset.src = args_dict["data_src"]
        hparams.data.train_split = args_dict["train_split"]
        hparams.data.validation_split = args_dict["validation_split"]
        hparams.data.batch_size = args_dict["batch_size"]
        hparams.data.num_workers = 0

        ## Add Normalization for Energy
        # hparams.model.normalizers = {
        #     "energy": [
        #         MC.PerAtomReferencingNormalizerConfig(
        #             per_atom_references=Path("./data/energy_reference.json")
        #         )
        #     ]
        # }

        ## Trainer Hyperparameters
        hparams.trainer = MC.TrainerConfig.draft()
        hparams.trainer.max_epochs = args_dict["max_epochs"]
        hparams.trainer.accelerator = "gpu"
        hparams.trainer.devices = args_dict["devices"]
        hparams.trainer.gradient_clip_algorithm = "value"
        hparams.trainer.gradient_clip_val = 1.0
        hparams.trainer.precision = "32"

        # Configure Model Checkpoint
        hparams.trainer.checkpoint = MC.ModelCheckpointConfig(
            monitor="val/total_loss",
            dirpath="./checkpoints-silica",
            filename="m3gnet-best",
            save_top_k=1,
            mode="min",
            every_n_epochs=10,
        )

        # Configure Logger
        hparams.trainer.loggers = [
            WandbLoggerConfig(
                project="MatterTune-Examples",
                name="M3GNet-silica",
            )
        ]

        # Additional trainer settings
        hparams.trainer.additional_trainer_kwargs = {
            "inference_mode": False,
        }

        hparams = hparams.finalize(strict=False)
        return hparams

    mt_config = hparams()
    model, trainer = MatterTuner(mt_config).tune()
    # trainer.save_checkpoint("finetuned.ckpt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="M3GNet-MP-2021.2.8-PES")
    parser.add_argument("--data-src", type=str)
    parser.add_argument("--task", type=str, default="silica")
    parser.add_argument("--energy-attr", type=str, default="y")
    parser.add_argument("--forces-attr", type=str, default="forces")
    parser.add_argument("--stress-attr", type=str, default="stress")
    
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--validation_split", type=float, default=0.05)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    args_dict = vars(args)
    main(args_dict)
