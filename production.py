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

def production(
    coords: Topology,
    forcefield: ForceField,
    output_state_data_filename = "production_state_data.csv",
    output_pdb_filename = "production_output.pdb",
    temperature: Quantity = 300*kelvin,
    friction_coeff: Quantity = 1/femtosecond,
    step_size: Quantity = 4*femtoseconds,
    duration: Quantity = 1*nanoseconds,
    steps_per_saved_frame: int = 1000
):
    print("Initialising production run...")
    
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
        temperature = True,
        potentialEnergy = True,
        speed=True
    )
    simulation.reporters.append(state_reporter)
    simulation.reporters.append(PDBReporter(output_pdb_filename, steps_per_saved_frame))

    # Production run  
    print("Running production...")
    simulation.step(int(duration / step_size))
    print("Done")
    return simulation

#########################################
# LOAD AND EQUILIBRATE
#########################################

valid_ffs = ['ani', 'amber']

parser = argparse.ArgumentParser(description='Production run an equilibrated peptide.')
parser.add_argument("pdb", help="Peptide PDB file")
parser.add_argument("ff", help=f"Forcefield/Potential to use: {valid_ffs}")

args = parser.parse_args()

TARGET_PDB = args.pdb
FORCEFIELD = args.ff.lower()

if FORCEFIELD not in ["ani", "amber"]:
    print(f"Invalid forcefield: {FORCEFIELD}, must be {valid_ffs}")
    quit()

# Load sample peptide
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
output_dir = f"equilibration_{pdb_name}_{FORCEFIELD}_{datetime.datetime.now().strftime('%H%M%S_%d%m%y')}"
os.mkdir(output_dir)
os.chdir(output_dir)

# Load pdb into modeller and add solvent
modeller = Modeller(pdb.topology, pdb.positions)
# modeller.addExtraParticles(forcefield)
modeller.addHydrogens(forcefield)
modeller.addSolvent(forcefield, model='tip3p', padding=1*nanometer, neutralize=False)

step_size = 4 * femtoseconds

# Equilibrate
simulation = equilibrate(
    modeller,
    forcefield,
    temp_range = range(0, 300, 20),
    time_per_temp_increment = 0.001 * nanoseconds,
    time_final_stage = 0.05 * nanoseconds,
    step_size = step_size,
)

simulation.saveState(f'{pdb_name}_{FORCEFIELD}_equilibrated.xml')

# Show graphs
report = pd.read_csv('equilibration_state_data.csv')
report = report.melt()

with sns.plotting_context('paper'): 
    g = sns.FacetGrid(data=report, row='variable', sharey=False )
    g.map(plt.plot, 'value')
    # format the labels with f-strings
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: f'{(x * step_size).value_in_unit(picoseconds):.1f}ps'))
    plt.savefig('equilibration.pdf', bbox_inches='tight')
    
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