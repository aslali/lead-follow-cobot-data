import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import analyze_utils as au
import seaborn as sns
import data_parameters as prm

ques = ['ad19', 'ad29', 'ad39']
all_ranks = [1, 2, 3]
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

def posthelp(df):
    posthelp_by_task(df)
    posthelp_by_difficulty(df)
    posthelp_by_pattern(df)


def posthelp_by_task(df):
    all_tasks = ['Task 1', 'Task 2', 'Task 3']
    nn = df.loc[:, ques]
    n1 = nn.rename(columns={ques[0]: all_tasks[0], ques[1]: all_tasks[1],
                       ques[2]: all_tasks[2]})

    fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    axs[0].hist(n1[all_tasks[0]])
    axs[1].hist(n1[all_tasks[1]])
    axs[2].hist(n1[all_tasks[2]])
    plt.show()

    n1n = pd.melt(n1.reset_index(), id_vars=['index'], var_name='Task')
    fig, ax = plt.subplots(1)
    sns.boxplot(y='value', x='Task', data=n1n, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data['value'] for _, group_data in n1n.groupby('Task')])
    print(result)
    if result.pvalue< 0.05:
        dunn_result = sp.posthoc_dunn(n1n, val_col='value', group_col='Task')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=n1n, dunn_result=dunn_result, fig_ax=ax, ly=6,
                                 pairs=significant_pairs, groups=all_tasks, x='Task', y='value', y_annotation=0)


    ax.set_ylim(0.8, 5.2)
    ax.set_xticklabels(['Task 1', 'Task 2', 'Task 3'])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([1, 2, 3, 4, 5])
    ax.set_ylabel('Reliance (Reverse Scale)')
    ax.set_xlabel('Tasks (Chronological order)')
    plt.tight_layout()
    plt.show()
    plt.show()


def posthelp_by_difficulty(df):
    var_name = 'Helpfulness'
    rtasks = []
    variable = []
    for i, row in df.iterrows():
        rank_fair = []
        rank_fair.append(row['RT1'])
        variable.append(row[ques[0]])

        rank_fair.append(row['RT2'])
        variable.append(row[ques[1]])

        rank_fair.append(row['RT3'])
        variable.append(row[ques[2]])
        rank_fair = stats.rankdata(rank_fair)
        rank_fair = [int(i) for i in rank_fair]
        rtasks += rank_fair
    ndf = pd.DataFrame({'Rank': rtasks, var_name: variable})

    fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    axs[0].hist(ndf[ndf['Rank'] == 1][var_name])
    axs[1].hist(ndf[ndf['Rank'] == 2][var_name])
    axs[2].hist(ndf[ndf['Rank'] == 3][var_name])
    plt.show()

    fig, ax = plt.subplots(1)
    sns.boxplot(y=var_name, x='Rank', data=ndf, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data[var_name] for _, group_data in ndf.groupby('Rank')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col=var_name, group_col='Rank')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=1.5,
                                 pairs=significant_pairs, groups=all_ranks, x='Rank', y=var_name, y_annotation=0)

    ax.set_ylim(0.8, 5.2)
    ax.set_xticklabels(['Rank 1', 'Rank 2', 'Rank 3'])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([1, 2, 3, 4, 5])
    ax.set_ylabel('Reliance (Reverse Scale)')
    ax.set_xlabel('Ranks')
    plt.tight_layout()
    plt.show()


def posthelp_by_pattern(df):
    var_name = 'Helpfulness'
    pattern = []
    variable = []
    for i, row in df.iterrows():
        pattern.append(prm.modes_patterns[row['mode']][0])
        variable.append(row[ques[0]])

        pattern.append(prm.modes_patterns[row['mode']][1])
        variable.append(row[ques[1]])

        pattern.append(prm.modes_patterns[row['mode']][2])
        variable.append(row[ques[2]])
    ndf = pd.DataFrame({var_name: variable, 'Pattern': pattern})

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(ndf[ndf['Pattern'] == 1][var_name])
    axs[0, 1].hist(ndf[ndf['Pattern'] == 2][var_name])
    axs[1, 0].hist(ndf[ndf['Pattern'] == 3][var_name])
    axs[1, 1].hist(ndf[ndf['Pattern'] == 4][var_name])
    plt.show()

    fig, ax = plt.subplots(1)
    sns.boxplot(y=var_name, x='Pattern', data=ndf, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data[var_name] for _, group_data in ndf.groupby('Pattern')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col=var_name, group_col='Pattern')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        ya = au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax,
                                  pairs=significant_pairs, groups=all_patterns, x='Pattern', y=var_name, y_annotation=0)
    ax.set_ylim(0, 5.2)
    ax.set_xticklabels(['Pattern 1', 'Pattern 2', 'Pattern 3'])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels([0, 1, 2, 3, 4, 5])
    ax.set_ylabel('Reliance (Reverse Scale)')
    ax.set_xlabel('Patterns')
    plt.tight_layout()
    plt.show()
    plt.show()
