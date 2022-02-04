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

# Equilibration function - Constant pressure & temp

def equilibrate(
    coords: Topology,
    forcefield: ForceField,
    final_pressure: Quantity = 1*atmosphere,
    temp_range: range = range(0, 300, 25),
    output_state_data_filename = "equilibration_state_data.csv",
    friction_coeff: Quantity = 1/femtosecond,
    step_size: Quantity = 4*femtoseconds,
    time_per_temp_increment: Quantity = 0.005*nanoseconds,
    time_final_stage: Quantity = 0.05*nanoseconds,
):
    print("Initialising equilibration run...")
    # adjust the range to include the highest temp (stop value)
    inclusive_temp_range = range(
        temp_range.start,
        temp_range.stop + temp_range.step,
        temp_range.step
    )
    temperatures = Quantity(inclusive_temp_range, kelvin)
    steps_per_temp_increment = int(time_per_temp_increment / step_size)
    steps_final_stage = int(time_final_stage / step_size)

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
        temperatures.min(),
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
        steps_per_temp_increment//10,
        temperature = True,
        potentialEnergy = True,
    )
    simulation.reporters.append(state_reporter)

    # Local energy minimisation
    print("Local energy minimisation...")
    simulation.minimizeEnergy()
    # Heating to final temp
    print(f"Equilibrating {temperatures.min()} to {temperatures.max()} in {len(temperatures)} stages, {time_per_temp_increment} per stage")
    for stage, temperature in enumerate(temperatures):
        print(f"Heating stage {stage+1}/{len(temperatures)} at {temperature}")
        integrator.setTemperature(temperature)
        simulation.step(steps_per_temp_increment)
    # Final equilibration, constant pressure 
    print(f"Final equilibration at {final_pressure} for {time_final_stage}")
    barostat = MonteCarloBarostat(
        final_pressure,
        temperatures.max()
    )
    system.addForce(barostat)
    simulation.step(steps_final_stage)
    print("Done")
    return simulation

#########################################
# LOAD AND EQUILIBRATE
#########################################

valid_ffs = ['ani', 'amber']

parser = argparse.ArgumentParser(description='Equilibrate a peptide.')
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
pdb.topology.setPeriodicBoxVectors(None)

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
os.makedirs(os.path.join("outputs", output_dir))
os.chdir(os.path.join("outputs", output_dir))

# Load pdb into modeller and add solvent
modeller = Modeller(pdb.topology, pdb.positions)
# modeller.addExtraParticles(forcefield)
modeller.addHydrogens(forcefield)
modeller.addSolvent(forcefield, model='tip3p', neutralize=False)

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

# Save as xml and pdb
simulation.saveState(f'{pdb_name}_{FORCEFIELD}_equilibrated.xml')
pdb.writeFile(
    simulation.topology, 
    simulation.context.getState(getPositions=True).getPositions(),
    open(f'{pdb_name}_{FORCEFIELD}_equilibrated.pdb', "w")
)

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