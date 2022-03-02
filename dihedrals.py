# Based on: http://archive.ambermd.org/201304/0256.html
# https://github.com/fylinhub/2D-free-energy-plots-and-similarity-calculation/blob/master/plot_freeenergy_chi1chi2.ipynb

# import MDAnalysis as mda
# from MDAnalysis.analysis.dihedrals import Ramachandran
import mdtraj as md
import matplotlib.pyplot as plt
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

output_dir = os.path.join(args.prod_dir, f"dihedral_analysis_{datetime.datetime.now().strftime('%H%M%S_%d%m%y')}")
os.mkdir(output_dir)

# def ramachandran(R):
#     # Plot Ramachandran
#     R.plot(
#         ax=plt.gca(), 
#         color = 'black', 
#         marker = ".", 
#         # s = (1/fig.dpi), 
#         ref = True # Set to false to hide blue reference heatmap of ramachandran plot
#     )



def free_energy(phi, psi):
    # Plot free energy
    x = phi
    y = psi

    degrees_fmt = lambda x, _: f"{x}°"
    ticks = np.arange(-180, 181, 60)
    temperature = 300
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=90, normed=False, range=np.array([[-180, 180], [-180,180]]))
    heatmap = heatmap.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # delta G = RT*ln(P)
    with np.errstate(divide='ignore'):
        heatmap = np.log(heatmap/heatmap.sum()) * (8.314/4.184) * temperature * -0.001
    heatmap[np.isinf(heatmap)] = np.nan
    plt.imshow(heatmap, extent=extent, origin='lower', interpolation=None, cmap='gist_earth')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(degrees_fmt))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(degrees_fmt))
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(color="black", alpha=0.2, linewidth=0.3, linestyle="--")
    plt.colorbar(label=r'$\Delta\mathit{G}$ (kcal mol$^{-1}$)')
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\psi$')

def timetrace(phi, psi):
    # Plot Phi and Psi
    angles = (phi, psi)
    fig, axs = plt.subplots(2,1,sharex=True, dpi=500)
    ticks = np.arange(-180, 181, 90)
    degrees_fmt = lambda x, _: f"{x}°"
    x = np.arange(len(psi))
    for j, name in enumerate((r'$\phi$', r'$\psi$')):
        ax = axs[j]
        ax.scatter(x, angles[j], marker=".", s=0.1)
        ax.grid(color="black", alpha=0.2, linewidth=0.3, linestyle="--")
        ax.set_yticks(ticks)
        ax.set(ylabel=name, ylim=(-180, 180))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(degrees_fmt))
    plt.xlabel('step')

plotters = {
    # "Ramachandran": ramachandran,
    "Free Energy Surface": free_energy,
    "Timetrace": timetrace
}

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

top = md.load(TOP).topology
# # we need to iterate the backbone in groups of four atoms, going from alpha carbon to alpha carbon
# # first select the backbone
# backbone = [atom for atom in top.atoms if (atom.is_backbone and atom.name != "O")]
# # then group into dihedral quartets 
# dihedral_atomgroups = [backbone[idx:idx+4] for idx, atom in enumerate(backbone) if atom.name in ("C", "N")]
# # clean up, removing the final atomgroup that doesn't have four atoms in it
# dihedral_atomgroups = [group for group in dihedral_atomgroups if len(group) == 4]
# # turn it into np array of indices (as mdtraj wants it)
# indices = np.array([[atom.index for atom in group] for group in dihedral_atomgroups])
# print(indices)

# Test there's enough RAM before starting analysis
print("Attempting to preallocate result array")

# use n-1 for uncapped
# n_dihedrals = top.n_residues - 1
# use n-2 for capped
n_dihedrals = top.n_residues - 2

phis = np.zeros((total_frames, n_dihedrals), dtype=float)
psis = np.zeros((total_frames, n_dihedrals), dtype=float)
print("Success, there should be enough RAM")

print(f"Starting...")
time_start = time.time()
chunk_size = 50000
traj = md.iterload(TRAJ, top=TOP, chunk=chunk_size)
for i, chunk in enumerate(traj):
    _, chunk_phis = md.compute_phi(chunk)
    _, chunk_psis = md.compute_psi(chunk)

    chunk_size = len(chunk)
    # res = md.compute_dihedrals(chunk, indices)
    phis[i*chunk_size:(i+1)*chunk_size, :] = np.rad2deg(chunk_phis)
    psis[i*chunk_size:(i+1)*chunk_size, :] = np.rad2deg(chunk_psis)
    speed = chunk_size // (time.time() - time_start)
    time_start = time.time()
    frames_remaining = total_frames - (i * chunk_size)
    print(f"{i*100*chunk_size/total_frames:.1f}%, {speed:.1f} frames per sec, {frames_remaining} frames remaining                 ", end="\r")
print("Dihedral analysis complete")

# Can manually shoehorn the data into particular orientations here
# phis *= -1
# psis *= -1

residues = tuple(top.residues)

for phi, psi, res1, res2 in zip(phis.T, psis.T, residues, residues[1:]):
    for title, plotter in plotters.items():
        plt.figure(0, facecolor="white", dpi=500)
        plt.title(f"{title} [dihedral {res1}-{res2}]")
        ax = plt.gca()
        ax.set_aspect(1)
        plotter(phi, psi)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{title}_{res1}_{res2}.png"))
        plt.clf()
        print(f"Saved {res1}-{res2} {title}")

print("Done")