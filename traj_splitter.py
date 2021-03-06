# Splits trajectory into multiple sub-trajectories, aligning the peptide within each short traj.

import mdtraj as md
import matplotlib.pyplot as plt
import time
import argparse
import numpy as np
import os.path
import datetime
from openmm import unit

TRAJECTORY_FN = "trajectory_reimaged.dcd" 
TOPOLOGY_FN = "topology.pdb"

# basic quantity string parsing ("1.2ns" -> openmm.Quantity)
unit_labels = {
    "us": unit.microseconds,
    "ns": unit.nanoseconds,
    "ps": unit.picoseconds,
    "fs": unit.femtoseconds
}

def parse_quantity(s):
    try:
        u = s.lstrip('0123456789.')
        v = s[:-len(u)]
        return unit.Quantity(
            float(v),
            unit_labels[u]
        )
    except Exception:
        raise ValueError(f"Invalid quantity: {s}")

parser = argparse.ArgumentParser()
parser.add_argument("prod_dir", help="Production directory to split trajectory into chunks")
parser.add_argument("-l", "--length", default="250ns", help="Length of each small trajectory")
parser.add_argument("-sf", "--savefreq", default="1ps", help="Timestep between each frame of the trajectory")
parser.add_argument("-f", "--format", default="dcd", help="Format to save trajectories. Must be supported by MDTraj.")
args = parser.parse_args()

LENGTH = parse_quantity(args.length)
SAVEFREQ = parse_quantity(args.savefreq)
FRAMES_PER_SUBTRAJ = int(LENGTH/SAVEFREQ)

if not os.path.isdir(args.prod_dir):
    print(f"Production directory to resume is not a directory: {args.prod_dir}")
    quit()

# Check all required files exist in prod directory to resume
files_available = os.listdir(args.prod_dir)
files_required = (
    TOPOLOGY_FN,
    TRAJECTORY_FN,
)

if not all(filename in files_available for filename in files_required):
    print(f"Production directory to analyse must contain files with the following names: {files_required}")
    quit()

output_dir = os.path.join(args.prod_dir, f"trajectory_split_{datetime.datetime.now().strftime('%H%M%S_%d%m%y')}")
os.mkdir(output_dir)

TRAJ = os.path.join(args.prod_dir, TRAJECTORY_FN)
TOP = os.path.join(args.prod_dir, TOPOLOGY_FN)

print("Initialising...")
with md.formats.DCDTrajectoryFile(TRAJ, mode="r") as dcd:
    total_frames = len(dcd)
print(f"{total_frames} frames total")

top = md.load(TOP).topology
# heavy_atoms = top.select("symbol != H")
heavy_atoms = top.select("protein")

print(f"Starting...")
time_start = time.time()
traj = md.iterload(TRAJ, top=TOP, chunk=FRAMES_PER_SUBTRAJ, atom_indices=heavy_atoms)
reference=None
chunk_xyzs = []
chunk_dihedrals = []

# # dumb hack to make sure the NHMe terminus is properly connected to the chain so i can reimage the molecule for ani properly
# atoms = list(top.atoms)
# top.add_bond(atoms[28], atoms[37])
# sorted_bonds = sorted(top.bonds, key=lambda bond: bond[0].index)
# sorted_bonds = np.asarray([[b0.index, b1.index] for b0, b1 in sorted_bonds], dtype=np.int32)

for i, chunk in enumerate(traj):
    if not reference:
        reference = chunk
    chunk = chunk.superpose(
        reference
    )

    # this is not ideal memory usage and might cause problems for large trajectories.
    # we're iteratively loading and processing a trajectory, just to accumulate it all in memory then dump into a .npz
    # Ideally msm analysis should be able to open a collection of short trajs iteratively, rather than from a single .npz archve
    chunk_xyzs.append(chunk.xyz.reshape(chunk.xyz.shape[0], -1))
    dihedrals = np.zeros((len(chunk), 2, top.n_residues - 2))
    md.compute_phi(chunk)
    dihedrals[:, 0, :] = md.compute_phi(chunk)[1]
    dihedrals[:, 1, :] = md.compute_psi(chunk)[1]
    chunk_dihedrals.append(dihedrals)

    subtraj_savename = os.path.join(output_dir, f"{i}.{args.format}")
    with md.open(subtraj_savename, mode="w") as fh:
        fh.write(
            chunk.xyz*10, 
            cell_lengths = chunk.unitcell_lengths*10, 
            cell_angles = chunk.unitcell_angles*10
        )

    speed = FRAMES_PER_SUBTRAJ // (time.time() - time_start)
    time_start = time.time()
    frames_remaining = total_frames - (i * FRAMES_PER_SUBTRAJ)
    print(f"{i*100*FRAMES_PER_SUBTRAJ/total_frames:.1f}%, {speed:.1f} frames per sec, {frames_remaining} frames remaining                 ", end="\r")
print("Trajectory splitting complete" + " "*40)

print("Creating .NPZ archive...")

# this is gross but it works
chunk_xyzs = np.vstack(chunk_xyzs)
chunk_dihedrals = np.vstack(chunk_dihedrals)
print(chunk_xyzs.shape)
print(chunk_dihedrals.shape)

np.savez(
    os.path.join(output_dir, f"coords_archive"), 
    chunk_xyzs
)
np.savez(
    os.path.join(output_dir, f"dihedrals_archive"), 
    chunk_dihedrals
)

print("Done")