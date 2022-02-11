import mdtraj as md
import time

TRAJ = "outputs/production_aaa_amber_amber_160126_260122/production_output_test.dcd"
TOP = "pdbs_equilibrated/aaa_amber.pdb"
OUT = "traj_unsolvated.dcd"

print("Initialising...")
stride = 10000
t = md.iterload(TRAJ, top=TOP, chunk=1, stride=stride)
total_frames = 0
for _ in t:
    total_frames += 1
    print(f"Counting chunks... {total_frames}   ", end="\r")
total_frames *= stride
print(f"{total_frames} frames total             ")

print("Starting... ")
outfile = md.formats.DCDTrajectoryFile(OUT, "w")
chunk_size = 500
traj = md.iterload(TRAJ, top=TOP, chunk=chunk_size)

time_start = time.time()
for i, chunk in enumerate(traj):
    chunk = chunk.remove_solvent()
    outfile.write(
        chunk.xyz, 
        cell_lengths = chunk.unitcell_lengths, 
        cell_angles = chunk.unitcell_angles)
    speed = chunk_size // (time.time() - time_start)
    time_start = time.time()
    frames_remaining = total_frames - (i * chunk_size)
    print(f"{i*100*chunk_size/total_frames:.1f}%, {speed:.1f} frames per sec, {frames_remaining} frames remaining                 ", end="\r")

outfile.close()
print(f"\n\nDone, saved to {OUT}")