import mdtraj as md
import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime

TRAJECTORY_FN = "trajectory.dcd" 
TOPOLOGY_FN = "topology.pdb"

parser = argparse.ArgumentParser()
parser.add_argument("prod_dir", help="Production directory to perform analysis on")
args = parser.parse_args()

if not os.path.isdir(args.prod_dir):
    print(f"Production directory is not a directory: {args.prod_dir}")
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

output_dir = os.path.join(args.prod_dir, f"terminal_distance_analysis_{datetime.datetime.now().strftime('%H%M%S_%d%m%y')}")
os.mkdir(output_dir)

TRAJ = os.path.join(args.prod_dir, TRAJECTORY_FN)
TOP = os.path.join(args.prod_dir, TOPOLOGY_FN)

print("Initialising...")
stride = 10000
t = md.iterload(TRAJ, top=TOP, chunk=1, stride=stride)
total_frames = 0
for _ in t:
    total_frames += 1
    print(f"Counting chunks... {total_frames}   ", end="\r")
total_frames *= stride
print(f"{total_frames} frames total             ")

# Test there's enough RAM before starting analysis
print("Attempting to preallocate result array")
results = np.zeros(total_frames, dtype=float)
print("Success, there should be enough RAM")

print("Starting... ")
top = md.load(TOP).topology
alpha_carbons = top.select("name CA")
terminal_carbons = np.array((alpha_carbons[0], alpha_carbons[-1])).reshape((1,2))
chunk_size = 500
traj = md.iterload(TRAJ, top=TOP, chunk=chunk_size)
time_start = time.time()
for i, chunk in enumerate(traj):
    cl = len(chunk)
    results[i*cl:(i+1)*cl] = md.compute_distances(
        chunk, 
        periodic=True, 
        atom_pairs=terminal_carbons
    )[:, 0]
    speed = chunk_size // (time.time() - time_start)
    time_start = time.time()
    frames_remaining = total_frames - (i * chunk_size)
    print(f"{i*100*chunk_size/total_frames:.1f}%, {speed:.1f} frames per sec, {frames_remaining} frames remaining                 ", end="\r")


plt.figure(0, facecolor="white", dpi=500)
plt.title("Terminal Carbon Distance")
ax = plt.gca()
# ax.set_aspect(1)
ax.grid(color="black", alpha=0.2, linewidth=0.3, linestyle="--")
ax.plot(results, linewidth=0.1)
ax.set(xlabel="Step", ylabel="Distance / nm")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "terminal_carbons.png"))
plt.clf()
print(f"Saved to {output_dir}")

# fig, axs = plt.subplots(2,1,sharex=True, dpi=500)
# ticks = np.arange(-180, 181, 90)
# degrees_fmt = lambda x, _: f"{x}°"
# for j, name in enumerate((r'$\psi$', r'$\phi$')):
#     ax = axs[j]
#     ax.plot(R.angles[:, 0, j], linewidth=0.1)
#     ax.grid(color="black", alpha=0.2, linewidth=0.3, linestyle="--")
#     ax.set_yticks(ticks)
#     ax.set(ylabel=name, ylim=(-180, 180))
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(degrees_fmt))
# plt.xlabel('step')