import mdtraj as md
import time
import numpy as np

"""
A simple way to get conformations from a trajectory.
Provide phi and psi angle pairs to get PDBs of the molecule in these conformations.
These should correspond to energy wells on the free energy surface.
You are much better at peak picking than any algorithm I could write.
"""

TRAJ = "outputs/production_aaa_capped_amber_equilibrated_amber_112650_010322/trajectory.dcd"
TOP = "outputs/production_aaa_capped_amber_equilibrated_amber_112650_010322/topology.pdb"
OUT = "outputs/aaa_conformations"

# phi and psi angles. we'll search for angles with a tolerance about +-5 deg to make sure we find something
angle_pairs = (
    (-150, 150),
    (-65, 150),
    (-65, -30),
    (-150, -30),
    (55, 30)
)

pairs = np.round(np.deg2rad(angle_pairs), 1)

found_pairs = np.zeros(len(angle_pairs))

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
outfile = md.formats.PDBTrajectoryFile(OUT, "w")
chunk_size = 500
reference=None
traj = md.iterload(TRAJ, top=TOP, chunk=chunk_size)

time_start = time.time()
for i, chunk in enumerate(traj):
    if not reference:
        reference = chunk
    chunk = chunk.superpose(
        reference
    )
    _, chunk_phis = md.compute_phi(chunk)
    _, chunk_psis = md.compute_psi(chunk)

    # print(dihedrals.shape)

    
    for phi, psi in pairs:
        dihedrals = np.hstack((chunk_phis, chunk_psis))
        np.round(dihedrals, 1, dihedrals)
        dihedrals[:, :3] -= phi
        dihedrals[:, 3:] -= psi
        match_dihedrals = np.where(~np.any(dihedrals, axis=1))[0]
        # and ~np.any(chunk_psis - psi, axis=1)

        if match_dihedrals.size > 0:
            print(f"Found match for pair {phi}, {psi}: idx {match_dihedrals}")

    # outfile.write(
    #     chunk.xyz, 
    #     cell_lengths = chunk.unitcell_lengths, 
    #     cell_angles = chunk.unitcell_angles)
    speed = chunk_size // (time.time() - time_start)
    time_start = time.time()
    frames_remaining = total_frames - (i * chunk_size)
    print(f"{i*100*chunk_size/total_frames:.1f}%, {speed:.1f} frames per sec, {frames_remaining} frames remaining                 ", end="\r")

outfile.close()
print(f"\n\nDone, saved to {OUT}")