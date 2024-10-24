import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import analyze_utils as au
import scipy.stats as stats
import scikit_posthocs as sp
import data_parameters as prm
from matplotlib.patches import Rectangle

ques_conf = ['pre0', 'pre11', 'pre21', 'pre31']
ques_reliance = ['ad19', 'ad29', 'ad39']
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


def prepare_trust_data(df):
    ndf = pd.DataFrame()
    ndf['Trust 0'] = df.loc[:, 't01':'t04'].astype(float).mean(axis=1)
    ndf['Trust 1'] = df.loc[:, 't11':'t14'].astype(float).mean(axis=1)
    ndf['Trust 2'] = df.loc[:, 't21':'t24'].astype(float).mean(axis=1)
    ndf['Trust 3'] = df.loc[:, 't31':'t34'].astype(float).mean(axis=1)

    all_conf = ['Confidence 0', 'Confidence 1', 'Confidence 2', 'Confidence 3']
    df = df.rename(columns={ques_conf[0]: all_conf[0], ques_conf[1]: all_conf[1],
                            ques_conf[2]: all_conf[2], ques_conf[3]: all_conf[3]})

    ndf['Confidence 0'] = df['Confidence 0']
    ndf['Confidence 1'] = df['Confidence 1']
    ndf['Confidence 2'] = df['Confidence 2']
    ndf['Confidence 3'] = df['Confidence 3']

    ndf['dCT1'] = ndf['Confidence 1'].div(2) - ndf['Trust 0']
    ndf['dCT2'] = ndf['Confidence 2'].div(2) - ndf['Trust 1']
    ndf['dCT3'] = ndf['Confidence 3'].div(2) - ndf['Trust 2']

    ndf['Reliance 1'] = df[ques_reliance[0]]
    ndf['Reliance 2'] = df[ques_reliance[1]]
    ndf['Reliance 3'] = df[ques_reliance[2]]
    ndf['mode'] = df['mode']
    # for i, row in df.iterrows():
    #     ndf.loc[i, 'Pattern'] = int(prm.modes_patterns[row['mode']][0])

    ndf2 = ndf[['dCT1', 'dCT2', 'dCT3', 'Reliance 1', 'Reliance 2', 'Reliance 3', 'mode']]
    ndf3 = pd.melt(ndf2.reset_index(), id_vars=['Reliance 1', 'Reliance 2', 'Reliance 3', 'mode'],
                   var_name='var', value_vars=['dCT1', 'dCT2', 'dCT3'])
    ndf3['Task'] = ''
    ndf3['Pattern'] = ''
    ndf3['First_Pattern'] = ''
    for i, row in ndf3.iterrows():
        if row['var'] == 'dCT1':
            ndf3.loc[i, 'Task'] = 'Task 1'
            ndf3.loc[i, 'Pattern'] = int(prm.modes_patterns[row['mode']][0])
            ndf3.loc[i, 'First_Pattern'] = int(prm.modes_patterns[row['mode']][0])
            ndf3.loc[i, 'Reliance'] = ndf3.loc[i, 'Reliance 1']
        elif row['var'] == 'dCT2':
            ndf3.loc[i, 'Task'] = 'Task 2'
            ndf3.loc[i, 'Pattern'] = int(prm.modes_patterns[row['mode']][1])
            ndf3.loc[i, 'First_Pattern'] = int(prm.modes_patterns[row['mode']][0])
            ndf3.loc[i, 'Reliance'] = ndf3.loc[i, 'Reliance 2']
        elif row['var'] == 'dCT3':
            ndf3.loc[i, 'Task'] = 'Task 3'
            ndf3.loc[i, 'Pattern'] = int(prm.modes_patterns[row['mode']][2])
            ndf3.loc[i, 'First_Pattern'] = int(prm.modes_patterns[row['mode']][0])
            ndf3.loc[i, 'Reliance'] = ndf3.loc[i, 'Reliance 3']
    ndf3.drop(['Reliance 1', 'Reliance 2', 'Reliance 3'], axis=1, inplace=True)

    fig, ax = plt.subplots(1)
    sns.boxplot(y='value', x='Task', data=ndf3, fliersize=3, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data['value'] for _, group_data in ndf3.groupby('Task')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf3, val_col='value', group_col='Task')
        print(dunn_result)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1)
    sns.boxplot(y='value', x='Pattern', data=ndf3, fliersize=3, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data['value'] for _, group_data in ndf3.groupby('Pattern')])
    print('t', result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf3, val_col='value', group_col='Pattern')
        print(dunn_result)
        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf3, dunn_result=dunn_result, fig_ax=ax, ly=1,
                                 pairs=significant_pairs, groups=[2, 3, 4], x='Pattern', y='value', y_annotation=0)
    ax.set_xticklabels(['Pattern B', 'Pattern C', 'Pattern D'])
    ax.set_ylim(-11, 11)
    plt.tight_layout()
    ax.set_yticks(list(range(-10, 11, 2)))
    ax.set_yticklabels(list(range(-10, 11, 2)))
    ax.set_ylabel('Relative Trust (SC - T)')
    ax.set_xlabel('Patterns')
    currentAxis = plt.gca()
    currentAxis.add_patch(Rectangle((-0.45, 7.6), 0.99, 3.2, facecolor="#E2E2E2"))
    plt.text(-0.4, 8, 'Trust: T\nSC: Self-confidence\nRelative Trust = SC - T')
    plt.savefig('result_relTrust_task.eps', format='eps')
    plt.show()

    # ndf4 = ndf3.loc[ndf3['Task'] == 'Task 1', :]
    # # print(ndf4.to_string)
    # fig, ax = plt.subplots(1)
    # sns.boxplot(y='Reliance', x='First_Pattern', data=ndf4, fliersize=3, showmeans=True, meanprops=meanprops, **props)
    #
    # result = stats.kruskal(*[group_data['Reliance'] for _, group_data in ndf4.groupby('First_Pattern')])
    # print(result)
    # if result.pvalue < 0.4:
    #     dunn_result = sp.posthoc_dunn(ndf4, val_col='Reliance', group_col='First_Pattern')
    #     print(dunn_result)
    # plt.show()

def self_confidence(df):
    # print(df[['id','pre0']].to_string())
    print(df.loc[df['pre0']<10, 'id'].to_string())
    confidence_by_task(df, ques=ques_conf, ylabel='Self-confidence', ymax=30)
    # confidence_by_difficulty(df, ques=ques_conf, var_name='Confidence', ylabel='Confidence', ymax=40)
    confidence_by_pattern(df, ques=ques_conf, var_name='Confidence', ylabel='Self-confidence', ymax=35)


def confidence_by_task(df, ques, ylabel='Value', ymax=130):
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
        au.plot_significant_dunn(data=n1n, dunn_result=dunn_result, fig_ax=ax, ly=2,
                                 pairs=significant_pairs, groups=all_tasks, x='Task', y='value', y_annotation=0)
    ax.set_ylim([-5, ymax])
    ax.set_xticklabels(['Task 0\n(CE: None)', 'Task 1 \n (CE: None)',
                        'Task 2 \n (CE: Task 1)', 'Task 3\n (CE: Tasks 1 & 2)'])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels([0, 5, 10, 15, 20])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Tasks  (Chronological order)')
    plt.text(-0.4, -3, 'CE: Collaboration Experience')
    plt.tight_layout()
    plt.savefig('result_confid_task.eps', format='eps')
    plt.show()


def confidence_by_difficulty(df, ques, var_name, ylabel='Value', ymax=130):
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
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=2,
                                 pairs=significant_pairs, groups=all_ranks, x='Rank', y=var_name, y_annotation=0)

    ax.set_ylim([-5, ymax])
    ax.set_xticklabels(['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4'])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels([0, 5, 10, 15, 20])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Ranks')
    plt.tight_layout()
    plt.show()


def confidence_by_pattern(df, ques, var_name, ylabel='Value', ymax=130):
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

    fig, ax = plt.subplots(1)
    sns.boxplot(y=var_name, x='Pattern', data=ndf, fliersize=3, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data[var_name] for _, group_data in ndf.groupby('Pattern')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col=var_name, group_col='Pattern')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        ya = au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=2,
                                      pairs=significant_pairs, groups=all_patterns, x='Pattern', y=var_name,
                                      y_annotation=0)
    ax.set_ylim([-2, ymax])
    ax.set_xticklabels(['Pattern A\n(Task 0)', 'Pattern B', 'Pattern C', 'Pattern D'])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels([0, 5, 10, 15, 20])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Patterns')
    plt.tight_layout()
    plt.savefig('result_confid_pattern.eps', format='eps')
    plt.show()
