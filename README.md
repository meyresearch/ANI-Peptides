# ANI-Peptides

A collection of scripts to equilibrate and perform production runs of peptides using ANI and AMBER

## Installation

Developed on Ubuntu 20.4 with Nvidia driver 510 and CUDA 11.6

Tested working on Ubuntu 20.4 Nvidia driver 495 and CUDA 11.5

Minimum CUDA 11.4 required

```
git clone https://github.com/meyresearch/ANI-Peptides && cd ./ANI-Peptides
conda env create -n ani_test -f environment.yml
git clone https://github.com/meyresearch/openmm-ml
pip install ./openmm-ml/.
```

Key dependencies:

*   **OpenMM**

 Base molecular simulation toolkit

*   **TorchANI**

 PyTorch Implementation of ANI

*   **PyTorch**

 TorchANI runs on the PyTorch machine learning framework 

*   **OpenMM-Torch**

 A plugin for OpenMM that allows PyTorch static computation graphs (TorchANI) to be used in OpenMM as a TorchForce object, an OpenMM Force class 

*   **OpenMM-ML**

 Implements TorchANI as an OpenMM TorchForce using OpenMM-Torch. The glue that brings everything together!

*   **MDAnalysis**

 Trajectory analysis library

*   **Seaborn, Matplotlib, Pandas**

 Data plotting and manipulation

## Usage

In general, you will want to modify the scripts `equilibration.py` and `production.py` to perform simulations in a consistent way, then execute the scripts on your pdbs using the available command line arguments

### Equilibration

Example: `python equilibrate.py pdbs/aaa.pdb amber`

This will perform equilibration of aaa.pdb using AMBER forcefield. For more options, modify equilibration.py script.

The results will be saved in `/outputs/equilibration_aaa_amber_xxxxxx_xxxxxx`, where `xxxxxx_xxxxxx` is a timestamp
- State data (csv)
- Graphs of state data (pdf)
- Save state of equilibrated system (xml)
- Save state of equilibrated system (pdb)

### Production

Example: `python production.py pdbs_equilibrated/aaa_amber.pdb amber`

This will perform a production run of the equilibrated aaa_amber.pdb using AMBER forcefield. For more options, modify production.py script.

The results will be saved in `/outputs/production_aaa_amber_xxxxxx_xxxxxx`, including
- State data (csv)
- Graphs of state data (pdf)
- Frames of the production run saved at regular intervals (dcd)
