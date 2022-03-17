import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Make some graphs
report = pd.read_csv(os.path.join(output_dir, STATE_DATA_FN))
report = report.melt()

with sns.plotting_context('paper'): 
    g = sns.FacetGrid(data=report, row='variable', sharey=False )
    g.map(plt.plot, 'value')
    # format the labels with f-strings
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: f'{(x * stepsize).value_in_unit(unit.nanoseconds):.1f}ns'))
    plt.savefig(os.path.join(output_dir, 'graphs.png'), bbox_inches='tight')
    