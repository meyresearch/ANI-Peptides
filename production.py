# Import libraries

from multiprocessing.context import ForkProcess
from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmml import MLPotential
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import argparse
import datetime
import os

#########################################
# DEFINITIONS
#########################################

# Production function - Constant pressure & volume

class StatePrinter:
    headers = ("Progress:", "Time Remaining:")
    def write(self, string):
        string = string.strip()
        print(string)

def PrintReporter(interval, total_steps):
    return StateDataReporter(
        StatePrinter(),
        interval,
        progress = True,
        remainingTime = True,
        totalSteps = total_steps,
    )

def production(
    coords: Topology,
    forcefield: ForceField,
    output_state_data_filename = "production_state_data.csv",
    output_dcd_filename = "production_output.dcd",
    temperature: Quantity = 300*kelvin,
    friction_coeff: Quantity = 1/femtosecond,
    step_size: Quantity = 4*femtoseconds,
    duration: Quantity = 1*nanoseconds,
    steps_per_saved_frame: int = 1000
):
    print("Initialising production run...")

    total_steps = int(duration / step_size)
    
    # Create system
    system = forcefield.createSystem(
        coords.topology, 
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer,
        constraints=AllBonds,
        hydrogenMass=4*amu,
    )
    # Create constant temp integrator
    integrator = LangevinMiddleIntegrator(
        temperature,
        friction_coeff,
        step_size
    )
    # Create simulation and set initial positions
    simulation = Simulation(
        coords.topology,
        system,
        integrator,
        # Platform.getPlatformByName("OpenCL" if FORCEFIELD == "ani" else "CUDA")
    )
    simulation.context.setPositions(coords.positions)
    state_reporter = StateDataReporter(
        output_state_data_filename,
        steps_per_saved_frame,
        step = True,
        time = True,
        temperature = True,
        potentialEnergy = True,
        kineticEnergy = True,
        totalEnergy = True,
    )
    simulation.reporters.append(PrintReporter(steps_per_saved_frame*10, total_steps))
    simulation.reporters.append(state_reporter)
    simulation.reporters.append(DCDReporter(output_dcd_filename, steps_per_saved_frame))

    # Production run  
    print("Running production...")
    simulation.step(total_steps)
    print("Done")
    return simulation

#########################################
# LOAD AND EQUILIBRATE
#########################################

valid_ffs = ['ani', 'amber']

parser = argparse.ArgumentParser(description='Production run for an equilibrated peptide.')
parser.add_argument("pdb", help="Peptide PDB file, should be solvated and equilibrated")
parser.add_argument("ff", help=f"Forcefield/Potential to use: {valid_ffs}")

args = parser.parse_args()

TARGET_PDB = args.pdb
FORCEFIELD = args.ff.lower()

if FORCEFIELD not in ["ani", "amber"]:
    print(f"Invalid forcefield: {FORCEFIELD}, must be {valid_ffs}")
    quit()

# Load peptide
pdb = PDBFile(TARGET_PDB)

if FORCEFIELD == "amber":
    # Create AMBER forcefield
    forcefield = ForceField(
        'amber14-all.xml',
        'amber14/tip3p.xml'
    )
elif FORCEFIELD == "ani":
    forcefield = MLPotential('ani2x')

# make directory to save equilibration data
pdb_name = os.path.splitext(os.path.basename(TARGET_PDB))[0]
output_dir = f"production_{pdb_name}_{FORCEFIELD}_{datetime.datetime.now().strftime('%H%M%S_%d%m%y')}"
os.makedirs(os.path.join("outputs", output_dir))
os.chdir(os.path.join("outputs", output_dir))

step_size = 4 * femtoseconds

# Equilibrate
simulation = production(
    pdb,
    forcefield,
    duration=1*microseconds
)

# Show graphs
report = pd.read_csv('production_state_data.csv')
report = report.melt()

with sns.plotting_context('paper'): 
    g = sns.FacetGrid(data=report, row='variable', sharey=False )
    g.map(plt.plot, 'value')
    # format the labels with f-strings
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: f'{(x * step_size).value_in_unit(nanoseconds):.1f}ns'))
    plt.savefig('production.pdf', bbox_inches='tight')
    
# ns/day (sanity check ~500ns/day)
# run for a day, see number of flips
# print a trajectory of the aaa dihedrals, counting the flips
# heatmap of phi and psi would be a good first analysis, use mdanalysis 
# aiming for https://docs.mdanalysis.org/1.1.0/documentation_pages/analysis/dihedrals.html
# number of events going between minima states
# "timetrace" - a plot of the dihedral over time (aim for 500ns)
# do this first, shows how often you go back and forth. one plot for each phi/psi angle
# four plots - for each set of pairs
# this gives two heatmap plots like in the documentation