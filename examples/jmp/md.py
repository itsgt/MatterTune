from ase import Atoms
from ase.io import read, write
import numpy as np
from mattertune.finetune.base import FinetuneModuleBase
from mattertune.potential import MatterTunePotential
from mattertune.ase.calculator import MatterTuneASECalculator
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def md(args_dict:dict):
    # Load Model from checkpoint
    model = FinetuneModuleBase.load_from_checkpoint(args_dict['checkpoint'])
    potential = MatterTunePotential(
        model=model, 
        accelator="gpu",
        devices=args_dict['gpus'],
        batch_size=1
    )
    
    # Setup ASE calculator
    atoms:Atoms = read(args_dict['structure'])
    calculator = MatterTuneASECalculator(potential=potential, stress_coeff=0.0) # stress_coeff=0.0 since Langevin dynamics does not 
    atoms.set_calculator(calculator)
    
    ## Run Langevin dynamics
    from ase.md import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units
    
    os.makedirs(args_dict['results_dir'], exist_ok=True)
    os.remove(os.path.join(args_dict['results_dir'], "traj.xyz"))
    # Set the momenta corresponding to T=300K
    MaxwellBoltzmannDistribution(atoms, args_dict['temperature_K'] * units.kB)
    dyn = Langevin(
        atoms=atoms,
        timestep=1 * units.fs,
        temperature_K=args_dict['temperature_K'],
        friction=0.02,
    )
    def write_xyz():
        write(
            os.path.join(args_dict['results_dir'], "traj.xyz"),
            atoms,
            append=True
        )
    dyn.attach(
        write_xyz,
        interval=1,
    )
    # Can directly run the dynamics with the number of steps
    # dyn.run(args_dict['num_ps'] * 1000)
    # Here we run the dynamics with a progress bar
    pbar = tqdm(total=args_dict['num_ps'] * 1000, desc=f"Langevin Dynamics under {args_dict['temperature_K']}K")
    for _ in range(args_dict['num_ps'] * 1000):
        dyn.run(1)
        pbar.update(1)
    pbar.close()
    
    # Plot the trajectory
    traj = read(os.path.join(args_dict['results_dir'], "traj.xyz"), ":")
    temps = [atoms.get_temperature() for atoms in traj]
    plt.plot(np.arange(0, len(temps), 1)/1000, temps)
    plt.xlabel("Time (ps)")
    plt.ylabel("Temperature (K)")
    ## Log scale for y-axis
    plt.yscale("log")
    ## Mark the target temperature
    plt.axhline(args_dict['temperature_K'], color="r", linestyle="--", label=f"{args_dict['temperature_K']}K")
    plt.savefig(os.path.join(args_dict['results_dir'], "temperature.png"))
    plt.close()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/water_ef/last.ckpt")
    parser.add_argument("--structure", type=str, default="../data/H2O.cif")
    parser.add_argument("--temperature_K", type=float, default=300)
    parser.add_argument("--num_ps", type=int, default=10)
    parser.add_argument("--results_dir", type=str, default="./md_results")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    
    md(vars(args))
    
    

    
    