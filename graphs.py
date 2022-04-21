import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import os
import openmm.unit as unit

output_dir = "/home/mbowley/ANI-Peptides/outputs/equilibration_aaa_capped_amber_121250_310322"
STATE_DATA_FN = "equilibration_state_data.csv"

# Make some graphs
report = pd.read_csv(os.path.join(output_dir, STATE_DATA_FN))
report = report.melt()

with sns.plotting_context('paper'): 
    g = sns.FacetGrid(data=report, row='variable', sharey=False )
    g.map(plt.plot, 'value')
    # format the labels with f-strings
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: f'{(x * 10*unit.femtoseconds).value_in_unit(unit.picoseconds):.1f}ns'))
    plt.savefig(os.path.join(output_dir, 'graphs.png'), bbox_inches='tight')
    