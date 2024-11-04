from __future__ import annotations

import logging
import nshutils as nu
import rich
import os
from pathlib import Path

import mattertune.configs as MC
from mattertune import MatterTuner

logging.basicConfig(level=logging.DEBUG)

nu.pretty()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def hparams():
    hparams = MC.MatterTunerConfig.draft()

    # Model hparams
    hparams.model = MC.M3GNetBackboneConfig.draft()
    hparams.model.ckpt_path = "M3GNet-MP-2021.2.8-PES"
    hparams.model.graph_computer = MC.M3GNetGraphComputerConfig.draft()
    hparams.model.optimizer = MC.AdamWConfig(lr=1.0e-4)
    
    ## Properties
    hparams.model.properties = []
    energy = MC.EnergyPropertyConfig(loss=MC.MAELossConfig())
    hparams.model.properties.append(energy)
    forces = MC.ForcesPropertyConfig(loss=MC.MAELossConfig(), conservative=False)
    hparams.model.properties.append(forces)
    stress = MC.StressesPropertyConfig(loss=MC.MAELossConfig(), conservative=False)
    hparams.model.properties.append(stress)
    
    ## Data hparams
    hparams.data = MC.PerSplitDataConfig.draft()
    hparams.data.train = MC.XYZDatasetConfig.draft()
    hparams.data.train.src = "./data/water_ef.xyz"
    hparams.data.batch_size = 128
    
    # Trainer hparams
    hparams.lightning_trainer_kwargs = {
        "max_epochs": 1, 
        "accelerator": "gpu", 
        "devices": [3],
        # "strategy": "ddp",
        "gradient_clip_algorithm": "value",
        "gradient_clip_val": 1.0,
        "precision": "32",
    }
    
    hparams = hparams.finalize(strict=False)
    return hparams
    
model = MatterTuner(hparams()).tune()

potential = model.potential()
print(potential)

calculator = model.ase_calculator()
print(calculator)

import ase

# Create a test periodic system
atoms = ase.Atoms(
    "H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True
)
print(atoms)

# Predict the properties
print(potential.predict([atoms], model.hparams.properties))

# Set the calculator
atoms.calc = calculator

# Calculate the energy
energy = atoms.get_potential_energy()
print(energy)

## Parallelized Prediction
atoms_1 = ase.Atoms(
    "H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True
)
atoms_2 = ase.Atoms(
    "H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], cell=[10, 10, 10], pbc=True
)
atoms = [atoms_1, atoms_2]
predictions = potential.predict(atoms, ["energy", "forces"])
print("ase.Atoms 1 energy:", predictions[0]["energy"])
print("ase.Atoms 1 forces:", predictions[0]["forces"])
print("ase.Atoms 2 energy:", predictions[1]["energy"])
print("ase.Atoms 2 forces:", predictions[1]["forces"])