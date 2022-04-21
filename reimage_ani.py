from weakref import ref
import mdtraj as md
import time
import argparse
import numpy as np
import os.path
import datetime

TRAJECTORY_FN = "trajectory.dcd" 
TOPOLOGY_FN = "topology.pdb"

parser = argparse.ArgumentParser()
parser.add_argument("prod_dir", help="Production directory to perform analysis on")
args = parser.parse_args()

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

TRAJ = os.path.join(args.prod_dir, TRAJECTORY_FN)
TOP = os.path.join(args.prod_dir, TOPOLOGY_FN)

output_fn = os.path.join(args.prod_dir, f"trajectory_reimaged.dcd")
print("Initialising...")
with md.formats.DCDTrajectoryFile(TRAJ, mode="r") as dcd:
    total_frames = len(dcd)
print(f"{total_frames} frames total")

print("Starting... ")
outfile = md.formats.DCDTrajectoryFile(output_fn, "w")
chunk_size = 5000
traj = md.iterload(TRAJ, top=TOP, chunk=chunk_size)

time_start = time.time()
reference = None
cell_size = None
for i, chunk in enumerate(traj):
    if not reference:
        reference = chunk[0]
    if not cell_size:
        cell_size = chunk.unitcell_lengths[0, 0]
    chunk.make_molecules_whole(inplace=True)
    chunk.center_coordinates()

    # make_molecules_whole can't handle the NHME cap properly, so it is often still stretched across the pbc
    # this code checks each frame to see if there are atoms too far from the centre, and removes them.
    # it'd be better to somehow translate the end cap across the pbc to the right place but i'm pressed for time here
    # the erroneous frames do need to be removed, or the vampnet will pick them up as a separate state.
    # this method is also slow
    # good_frames = np.ones(len(chunk), dtype=bool)
    # if args.check_bondlengths:
    #     for i, frame in enumerate(chunk):
    #         if abs(frame.xyz.max()) > 1 or abs(frame.xyz.min()) > 1:
    #             good_frames[i] = False
    
    # nvm, this does it perfectly. tuneable and way faster.
    chunk.xyz[chunk.xyz < -1] += cell_size
    chunk.xyz[chunk.xyz > 1] -= cell_size
    chunk.center_coordinates()
    chunk.superpose(reference)

    outfile.write(
        chunk.xyz,
        cell_lengths = chunk.unitcell_lengths, 
        cell_angles = chunk.unitcell_angles)
    speed = chunk_size // (time.time() - time_start)
    time_start = time.time()
    frames_remaining = total_frames - (i * chunk_size)
    print(f"{i*100*chunk_size/total_frames:.1f}%, {speed:.1f} frames per sec, {frames_remaining} frames remaining                 ", end="\r")

outfile.close()
print(f"\n\nDone, saved to {output_fn}")