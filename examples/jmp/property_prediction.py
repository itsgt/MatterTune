from ase import Atoms
from ase.io import read, write
from mattertune.finetune.base import FinetuneModuleBase
from mattertune.potential import MatterTunePotential
from mattertune.ase.calculator import MatterTuneASECalculator
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def property_prediction(args_dict:dict):
    ## Load Model from checkpoint
    model = FinetuneModuleBase.load_from_checkpoint(args_dict['checkpoint'])
    potential = MatterTunePotential(
        model=model, 
        accelator="gpu",
        devices=args_dict['gpus'],
        batch_size=1
    )
    supported_properties = potential.get_supported_properties()
    print(f"Supported properties: {supported_properties}")
    
    ## Setup ASE calculator
    ## Here target_name_map maps the output head target_name ("band_gap") to property name ("bandgap")
    calculator = MatterTuneASECalculator(potrntial=potential, stress_coeff=0.0, target_name_map={"band_gap": "bandgap"}) 
    
    # Predict the property
    os.makedirs(args_dict['results_dir'], exist_ok=True)
    atoms_list = read(args_dict['structures'], index=":")
    gt_band_gaps = []
    band_gaps = []
    pbar = tqdm(total=len(atoms_list), desc="Predicting properties")
    for atoms in atoms_list:
        gt_band_gaps.append(atoms.info["band_gap"])
        atoms.calc = calculator
        property_value = atoms.get_properties("bandgap") ## Here I intentionally used "bandgap" instead of "band_gap" to show how the target_name_map works
        band_gaps.append(property_value)
        pbar.update(1)
    pbar.close()
    
    plt.plot(gt_band_gaps, band_gaps, "o")
    limit_max = max(max(gt_band_gaps), max(band_gaps))
    limit_min = min(min(gt_band_gaps), min(band_gaps))
    plt.plot([limit_min, limit_max], [limit_min, limit_max], "k--")
    plt.xlabel("Ground Truth Band Gap (eV)")
    plt.ylabel("Predicted Band Gap (eV)")
    plt.title("Band Gap Prediction")
    plt.savefig(os.path.join(args_dict['results_dir'], "band_gap.png"))
    plt.close()
    
    average_error = sum([abs(gt - pred) for gt, pred in zip(gt_band_gaps, band_gaps)]) / len(gt_band_gaps)
    print(f"Average error: {average_error}eV")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/bandgap/last.ckpt", help="Path to the model checkpoint")
    parser.add_argument("--structures", type=str, default="../data/mp_gap_sample_1000.xyz", help="Path to the structures file")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="List of GPU ids")
    parser.add_argument("--results_dir", type=str, default="./property_results", help="Directory to save the results")
    args_dict = vars(parser.parse_args())
    property_prediction(args_dict)
    