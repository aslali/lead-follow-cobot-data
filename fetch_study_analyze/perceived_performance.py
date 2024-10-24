import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import analyze_utils as au
import seaborn as sns

ques_time = ['ad110', 'ad210', 'ad310']
ques_acc = ['ad111', 'ad211', 'ad311']
ques_deci = ['ad112', 'ad212', 'ad312']
ques_collab = ['ad18', 'ad28', 'ad38']
all_ranks = [1, 2, 3, 4]
all_patterns = [1, 2, 3, 4]
props = {
    'boxprops': {'edgecolor': 'none'},
    'medianprops': {'color': 'black'},
    'whiskerprops': {'color': 'blue'},
    'capprops': {'color': 'blue'}
}

meanprops = {"marker": "o",
             "markerfacecolor": "white",
             "markeredgecolor": "black",
             "markersize": "8"}


def performance(df):
    performance_by_task(df)


def performance_by_task(df):
    nn = df.loc[:, ques_time + ques_acc + ques_deci + ques_collab]
    n1n = pd.melt(nn.reset_index(), id_vars=['index'], var_name='ques')
    n1n['ptype'] = ''
    n1n['Task'] = ''
    for i, row in n1n.iterrows():
        if row['ques'] in ques_acc:
            n1n.loc[i, 'ptype'] = 'Accuracy'
        elif row['ques'] in ques_time:
            n1n.loc[i, 'ptype'] = 'Time'
        elif row['ques'] in ques_deci:
            n1n.loc[i, 'ptype'] = 'Reliability'
        elif row['ques'] in ques_collab:
            n1n.loc[i, 'ptype'] = 'Collaboration'


        if row['ques'] in [ques_time[0], ques_acc[0], ques_deci[0], ques_collab[0]]:
            n1n.loc[i, 'Task'] = 'Task 1'
        elif row['ques'] in [ques_time[1], ques_acc[1], ques_deci[1], ques_collab[1]]:
            n1n.loc[i, 'Task'] = 'Task 2'
        else:
            n1n.loc[i, 'Task'] = 'Task 3'

    n1n['value'] = 3*(n1n['value']-1)/2 + 1
    fig, ax = plt.subplots(1)
    custom_palette = {'Time': '#fd7f6f', 'Accuracy': '#7eb0d5', 'Reliability': '#b2e061', 'Collaboration': '#ffb55a'}
    g = sns.boxplot(y='value', x='Task', hue='ptype', data=n1n, showmeans=True, meanprops=meanprops, fliersize=2, palette=custom_palette,**props)
    g.legend(loc='upper left', fontsize=9, ncol=2)
    g.legend_.texts[0].set_text('Time (Reverse Scale)')
    ax.set_ylim([0.8, 8.5])
    ax.set_xticklabels(['Task 1', 'Task 2', 'Task 3'])
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels([1, 2, 3, 4, 5,6, 7])
    ax.set_ylabel('Perceived Performance')
    ax.set_xlabel('Tasks  (Chronological order)')
    plt.tight_layout()
    plt.savefig('result_trac.eps', format='eps')
    plt.show()
    # n2n = n1n[n1n.loc[:, 'ptype']=='Collaboration']
    # result = stats.kruskal(*[group_data['value'] for _, group_data in n2n.groupby('Task')])
    # print(result)