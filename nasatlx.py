import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import analyze_utils as au
import scipy.stats as stats
import scikit_posthocs as sp
import data_parameters as prm

ques_mental = ['n01', 'n11', 'n21', 'n31']
ques_physical = ['n02', 'n12', 'n22', 'n32']
ques_temporal = ['n03', 'n13', 'n23', 'n33']
ques_performance = ['n04', 'n14', 'n24', 'n34']
ques_effort = ['n05', 'n15', 'n25', 'n35']
ques_frust = ['n06', 'n16', 'n26', 'n36']
all_ranks = [1, 2, 3, 4]
all_patterns = [1, 2, 3, 4]
props = {
    'boxprops': {'facecolor': 'none', 'edgecolor': 'red'},
    'medianprops': {'color': 'green'},
    'whiskerprops': {'color': 'blue'},
    'capprops': {'color': 'blue'}
}

meanprops = {"marker": "o",
             "markerfacecolor": "white",
             "markeredgecolor": "black",
             "markersize": "8"}


def tlx_mental(df):
    tlx_by_task(df, ques=ques_mental, ylabel='Mental Demand', ymax=140)
    # tlx_by_difficulty(df, ques=ques_mental, var_name='Mental', ylabel='Mental Demand', ymax=140)
    tlx_by_pattern(df, ques=ques_mental, var_name='Mental', ylabel='Mental Demand', ymax=150)


def tlx_physical(df):
    tlx_by_task(df, ques=ques_physical, ylabel='Physical Demand', ymax=110)
    # tlx_by_difficulty(df, ques=ques_physical, var_name='Physical', ylabel='Physical Demand', ymax=140)
    # tlx_by_pattern(df, ques=ques_physical, var_name='Physical', ylabel='Physical Demand', ymax=140)


def tlx_temporal(df):
    tlx_by_task(df, ques=ques_temporal, ylabel='Temporal Demand', ymax=110)
    # tlx_by_difficulty(df, ques=ques_temporal, var_name='Temporal', ylabel='Temporal Demand', ymax=140)
    # tlx_by_pattern(df, ques=ques_temporal, var_name='Temporal', ylabel='Temporal Demand', ymax=140)


def tlx_performance(df):
    tlx_by_task(df, ques=ques_performance, ylabel='Perceived Performance', ymax=140)
    # tlx_by_difficulty(df, ques=ques_performance, var_name='Performance', ylabel='Performance', ymax=110)
    # tlx_by_pattern(df, ques=ques_performance, var_name='Performance', ylabel='Performance', ymax=110)


def tlx_effort(df):
    tlx_by_task(df, ques=ques_effort, ylabel='Effort', ymax=140)
    # tlx_by_difficulty(df, ques=ques_effort, var_name='Effort', ylabel='Effort', ymax=140)
    tlx_by_pattern(df, ques=ques_effort, var_name='Effort', ylabel='Effort', ymax=140)


def tlx_frust(df):
    tlx_by_task(df, ques=ques_frust, ylabel='Frustration', ymax=140)
    # tlx_by_difficulty(df, ques=ques_frust, var_name='Frustration', ylabel='Frustration', ymax=140)
    tlx_by_pattern(df, ques=ques_frust, var_name='Frustration', ylabel='Frustration', ymax=140)


def tlx_by_task(df, ques, ylabel='Value', ymax=130):
    all_tasks = ['Task 0', 'Task 1', 'Task 2', 'Task 3']
    nn = df.loc[:, ques]
    n1 = nn.rename(columns={ques[0]: all_tasks[0], ques[1]: all_tasks[1],
                            ques[2]: all_tasks[2], ques[3]: all_tasks[3]})

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(n1[all_tasks[0]])
    axs[0, 1].hist(n1[all_tasks[1]])
    axs[1, 0].hist(n1[all_tasks[2]])
    axs[1, 1].hist(n1[all_tasks[3]])
    plt.show()

    n1n = pd.melt(n1.reset_index(), id_vars=['index'], var_name='Task')
    # print(n1n.to_string())
    fig, ax = plt.subplots(1)
    sns.boxplot(y='value', x='Task', data=n1n, fliersize=3, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data['value'] for _, group_data in n1n.groupby('Task')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(n1n, val_col='value', group_col='Task')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=n1n, dunn_result=dunn_result, fig_ax=ax, ly=6,
                                 pairs=significant_pairs, groups=all_tasks, x='Task', y='value', y_annotation=0)
    ax.set_ylim([-5, ymax])
    ax.set_xticklabels(['Task 0', 'Task 1', 'Task 2', 'Task 3'])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([0, 20, 40, 60, 80, 100])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Tasks  (Chronological order)')
    figname = 'result_' + ylabel + '_task.eps'
    plt.tight_layout()
    plt.savefig(figname, format='eps')
    plt.show()


def tlx_by_difficulty(df, ques, var_name, ylabel='Value', ymax=130):
    tasks = []
    rtasks = []
    variable = []
    for i, row in df.iterrows():
        rtasks.append(row['RT0'])
        variable.append(row[ques[0]])
        tasks.append('Task 0')

        rtasks.append(row['RT1'])
        variable.append(row[ques[1]])
        tasks.append('Task 1')

        rtasks.append(row['RT2'])
        variable.append(row[ques[2]])
        tasks.append('Task 2')

        rtasks.append(row['RT3'])
        variable.append(row[ques[3]])
        tasks.append('Task 3')
    ndf = pd.DataFrame({'Rank': rtasks, var_name: variable, 'Task': tasks})

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(ndf[ndf['Rank'] == 1][var_name])
    axs[0, 1].hist(ndf[ndf['Rank'] == 2][var_name])
    axs[1, 0].hist(ndf[ndf['Rank'] == 3][var_name])
    axs[1, 1].hist(ndf[ndf['Rank'] == 4][var_name])
    plt.show()

    fig, ax = plt.subplots(1)
    sns.boxplot(y=var_name, x='Rank', data=ndf, fliersize=3, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data[var_name] for _, group_data in ndf.groupby('Rank')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col=var_name, group_col='Rank')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=6,
                                 pairs=significant_pairs, groups=all_ranks, x='Rank', y=var_name, y_annotation=0)

    ax.set_ylim([-5, ymax])
    ax.set_xticklabels(['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4'])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([0, 20, 40, 60, 80, 100])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Ranks')
    plt.tight_layout()
    plt.show()


def tlx_by_pattern(df, ques, var_name, ylabel='Value', ymax=130):
    tasks = []
    pattern = []
    variable = []
    for i, row in df.iterrows():
        pattern.append(1)
        variable.append(row[ques[0]])
        tasks.append('Task 0')

        pattern.append(prm.modes_patterns[row['mode']][0])
        variable.append(row[ques[1]])
        tasks.append('Task 1')

        pattern.append(prm.modes_patterns[row['mode']][1])
        variable.append(row[ques[2]])
        tasks.append('Task 2')

        pattern.append(prm.modes_patterns[row['mode']][2])
        variable.append(row[ques[3]])
        tasks.append('Task 3')
    ndf = pd.DataFrame({'Task': tasks, var_name: variable, 'Pattern': pattern})

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(ndf[ndf['Pattern'] == 1][var_name])
    axs[0, 1].hist(ndf[ndf['Pattern'] == 2][var_name])
    axs[1, 0].hist(ndf[ndf['Pattern'] == 3][var_name])
    axs[1, 1].hist(ndf[ndf['Pattern'] == 4][var_name])
    plt.show()
    ndf = ndf[ndf['Task'] != 'Task 0']
    fig, ax = plt.subplots(1)
    sns.boxplot(y=var_name, x='Pattern', data=ndf, fliersize=3, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data[var_name] for _, group_data in ndf.groupby('Pattern')])
    print(result)
    if result.pvalue < 0.05:

        dunn_result = sp.posthoc_dunn(ndf, val_col=var_name, group_col='Pattern')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        ya = au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=6,
                                      pairs=significant_pairs, groups=all_patterns[1:], x='Pattern', y=var_name,
                                      y_annotation=0)
        # groups = all_patterns[1:]
    ax.set_ylim([-5, ymax])
    ax.set_xticklabels(['Pattern B', 'Pattern C', 'Pattern D'])
    # ax.set_xticklabels(['Pattern A', 'Pattern B', 'Pattern C', 'Pattern D'])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([0, 20, 40, 60, 80, 100])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Patterns')
    plt.tight_layout()
    figname = 'result_' + ylabel + '_pattern.eps'
    plt.savefig(figname, format='eps')
    plt.show()
