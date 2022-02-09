import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran
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

def ramachandran(R):
    # Plot Ramachandran
    R.plot(
        ax=plt.gca(), 
        color = 'black', 
        marker = ".", 
        # s = (1/fig.dpi), 
        ref = True # Set to false to hide blue reference heatmap of ramachandran plot
    )

def free_energy(R):
    # Plot free energy
    x = R.angles[:,0,0]
    y = R.angles[:,0,1]

    degrees_fmt = lambda x, _: f"{x}°"
    ticks = np.arange(-180, 181, 60)
    temperature = 300
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=90, normed=False, range=np.array([[-180, 180], [-180,180]]))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # delta G = RT*ln(P)
    heatmap = np.log(heatmap/heatmap.max()) * (8.314/4.184) * temperature * -0.001
    plt.imshow(heatmap.T, extent=extent, origin='lower', interpolation=None, cmap='gist_earth')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(degrees_fmt))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(degrees_fmt))
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.grid(color="black", alpha=0.2, linewidth=0.3, linestyle="--")
    plt.colorbar(label=r'$\Delta\mathit{G}$ (kcal mol$^{-1}$)')
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\psi$')

def timetrace(R):
    # Plot Phi and Psi
    fig, axs = plt.subplots(2,1,sharex=True, dpi=500)
    ticks = np.arange(-180, 181, 90)
    degrees_fmt = lambda x, _: f"{x}°"
    for j, name in enumerate((r'$\psi$', r'$\phi$')):
        ax = axs[j]
        ax.plot(R.angles[:, 0, j], linewidth=0.1)
        ax.grid(color="black", alpha=0.2, linewidth=0.3, linestyle="--")
        ax.set_yticks(ticks)
        ax.set(ylabel=name, ylim=(-180, 180))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(degrees_fmt))
    plt.xlabel('step')

plotters = {
    "Ramachandran": ramachandran,
    "Free Energy Surface": free_energy,
    "Timetrace": timetrace
}

u = mda.Universe(
    os.path.join(args.prod_dir, TOPOLOGY_FN), 
    os.path.join(args.prod_dir, TRAJECTORY_FN)
)
r = u.select_atoms("backbone")

num_res_pairs = len(r.residues) - 1
res_idxs = ((i, i+2) for i in range(num_res_pairs))
angles = (r.residues[i:j] for i,j in res_idxs)

print(f"Starting...")
time_start = time.time()
for i, angle in enumerate(angles):
    print(f"Analysing residue pair {i} of {num_res_pairs}...")
    R = Ramachandran(angle).run()

    for title, plotter in plotters.items():
        plt.figure(0, facecolor="white", dpi=500)
        plt.title(f"{title} [dihedral {i}-{i+1}]")
        ax = plt.gca()
        ax.set_aspect(1)
        # savename = os.path.splitext(os.path.basename(PDB))[0]
        plotter(R)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{title}_{i}_{i+1}.png"))
        plt.clf()
        print(f"Saved {title}")

print("Done")