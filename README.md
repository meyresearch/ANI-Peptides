# ANI-Peptides

A small collection of scripts to equilibrate and perform production runs of peptides using ANI and AMBER

## Installation

```
conda create -n ani 
conda activate ani
conda install -c conda-forge openmm openmm-torch pytorch torchani seaborn matplotlib pandas mdanalysis cudatoolkit=11.5
git clone https://github.com/yueyericardo/openmm-ml
pip install ./openmm-ml/.
```

What are these dependencies used for?

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