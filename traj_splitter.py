# Splits trajectory into multiple sub-trajectories, aligning the peptide within each short traj.

import mdtraj as md
import matplotlib.pyplot as plt
import time
import argparse
import numpy as np
import os.path
import datetime
from openmm import unit

TRAJECTORY_FN = "trajectory.dcd" 
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
stride = 10000
t = md.iterload(TRAJ, top=TOP, chunk=1, stride=stride)
total_frames = 0
for chunk in t:
    total_frames += 1
    print(f"Counting chunks... {total_frames}   ", end="\r")
total_frames *= stride
print(f"{total_frames} frames total             ")

top = md.load(TOP).topology
heavy_atoms = top.select("symbol != H")
heavy_atoms = top.select("protein")

print(f"Starting...")
time_start = time.time()
traj = md.iterload(TRAJ, top=TOP, chunk=FRAMES_PER_SUBTRAJ, atom_indices=heavy_atoms)
reference=None
chunk_xyzs = []
chunk_dihedrals = []

for i, chunk in enumerate(traj):
    if not reference:
        reference = chunk
    chunk = chunk.superpose(
        reference
    )

    # this is not ideal memory usage and might cause problems for large trajectories.
    # we're iteratively loading and processing a trajectory, just to accumulate it all in memory then dump into a .npz
    # Ideally msm analysis should be able to open a collection of short trajs iteratively, rather thank from a single .npz archve
    # chunk_xyzs.append(chunk.xyz)
    # chunk_dihedrals.append(md.compute_phi(chunk))
    # chunk_dihedrals.append(md.compute_psi())

    subtraj_savename = os.path.join(output_dir, f"{i}.{args.format}")
    with md.open(subtraj_savename, mode="w") as fh:
        fh.write(
            chunk.xyz, 
            cell_lengths = chunk.unitcell_lengths, 
            cell_angles = chunk.unitcell_angles
        )

    speed = FRAMES_PER_SUBTRAJ // (time.time() - time_start)
    time_start = time.time()
    frames_remaining = total_frames - (i * FRAMES_PER_SUBTRAJ)
    print(f"{i*100*FRAMES_PER_SUBTRAJ/total_frames:.1f}%, {speed:.1f} frames per sec, {frames_remaining} frames remaining                 ", end="\r")
print("Trajectory splitting complete" + " "*40)

print("Creating .NPZ archive...")
archive_savename = os.path.join(output_dir, f"traj_split_archive")
np.savez(archive_savename, *chunk_xyzs)

print("Done")