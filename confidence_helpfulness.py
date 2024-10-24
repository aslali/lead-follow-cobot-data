import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import analyze_utils as au

import scipy.stats as stats
import scikit_posthocs as sp
import data_parameters as prm

cases_conf = ['pre0', 'pre11', 'pre21', 'pre31']
cases_prehelp = ['pre12', 'pre22', 'pre32']
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

def conf_prehelp_all(df):
    all_tasks = ['Task 0', 'Task 1', 'Task 2', 'Task 3']

    df_conf_prehelp = df.loc[:, cases_conf + cases_prehelp]

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(df_conf_prehelp[cases_conf[0]])
    axs[0, 1].hist(df_conf_prehelp[cases_conf[1]])
    axs[1, 0].hist(df_conf_prehelp[cases_conf[2]])
    axs[1, 1].hist(df_conf_prehelp[cases_conf[3]])
    plt.show()

    fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    axs[0].hist(df_conf_prehelp[cases_prehelp[0]])
    axs[1].hist(df_conf_prehelp[cases_prehelp[1]])
    axs[2].hist(df_conf_prehelp[cases_prehelp[2]])
    plt.show()

    fig, ax = plt.subplots(tight_layout=False)
    ndff = pd.melt(df_conf_prehelp.reset_index(), id_vars=['index'])
    task_dic = {'pre0': 'Task 0', 'pre11': 'Task 1', 'pre21': 'Task 2', 'pre31': 'Task 3',
                'pre12': 'Task 1', 'pre22': 'Task 2', 'pre32': 'Task 3'}
    task_type = {'pre0': 'Conf', 'pre11': 'Conf', 'pre21': 'Conf', 'pre31': 'Conf',
                 'pre12': 'Help', 'pre22': 'Help', 'pre32': 'Help'}
    ndff['task'] = ''
    ndff['type'] = ''
    for i, row in ndff.iterrows():
        ndff.at[i, 'task'] = task_dic[row['variable']]
        ndff.at[i, 'type'] = task_type[row['variable']]

    meanprops = {"marker": "o",
                 "markerfacecolor": "white",
                 "markeredgecolor": "black",
                 "markersize": "6"}
    custom_palette = {'Conf': '#607D8B', 'Help': '#A3682A'}
    g = sns.boxplot(y='value', x='task', data=ndff, hue='type', fliersize=3, palette=custom_palette, showmeans=True,
                    meanprops=meanprops, )
    g.legend(loc='upper left', fontsize=9)
    g.legend_.texts[0].set_text('Self-Confidence')
    g.legend_.texts[1].set_text('Anticipated helpfulness')

    # sns.boxplot(data=ndf1, x=ndf1['variable'], color="red", ax=ax)

    df_conf = ndff[ndff['type'] == 'Conf']

    result = stats.kruskal(*[group_data['value'] for _, group_data in df_conf.groupby('task')])
    print(result)

    dunn_result = sp.posthoc_dunn(df_conf, val_col='value', group_col='task')
    print(dunn_result)
    significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    ya = au.plot_significant_dunn(data=df_conf, dunn_result=dunn_result, fig_ax=ax, xth=-0.25,
                                  pairs=significant_pairs, groups=all_tasks, x='task', y='value', y_annotation=0)

    #
    #
    df_help = ndff[ndff['type'] == 'Help']
    result = stats.kruskal(*[group_data['value'] for _, group_data in df_help.groupby('task')])
    print(result)
    dunn_result = sp.posthoc_dunn(df_help, val_col='value', group_col='task')
    print(dunn_result)
    significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    print(significant_pairs)
    au.plot_significant_dunn(data=df_help, dunn_result=dunn_result, fig_ax=ax, xth=0.25,
                             pairs=significant_pairs, groups=all_tasks, x='task', y='value', y_annotation=ya)

    ax.set_ylim(-0.99, 33)

    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels([0, 5, 10, 15, 20])
    ax.set_ylabel('Value')
    ax.set_xlabel('Tasks (Chronological order)')
    plt.tight_layout()
    plt.show()


def prehelp_by_pattern(df):
    var_name = 'Helpfulness'
    pattern = []
    variable = []
    for i, row in df.iterrows():
        pattern.append(prm.modes_patterns[row['mode']][0])
        variable.append(row[cases_prehelp[0]])

        pattern.append(prm.modes_patterns[row['mode']][1])
        variable.append(row[cases_prehelp[1]])

        pattern.append(prm.modes_patterns[row['mode']][2])
        variable.append(row[cases_prehelp[2]])
    ndf = pd.DataFrame({var_name: variable, 'Pattern': pattern})

    fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    axs[0].hist(ndf[ndf['Pattern'] == 2][var_name])
    axs[1].hist(ndf[ndf['Pattern'] == 3][var_name])
    axs[2].hist(ndf[ndf['Pattern'] == 4][var_name])
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
                                      pairs=significant_pairs, groups=all_patterns, x='Pattern', y=var_name,
                                      y_annotation=0)
    ax.set_ylim(0, 30)
    ax.set_xticklabels(['Pattern 1', 'Pattern 2', 'Pattern 3'])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_yticklabels([0, 1, 2, 3, 4, 5])
    ax.set_ylabel('Expected Helpfulness')
    ax.set_xlabel('Patterns')
    plt.tight_layout()
    plt.show()
    plt.show()


def prehelp_by_task(df):
    all_tasks = ['Task 1', 'Task 2', 'Task 3']
    nn = df.loc[:, cases_prehelp]
    n1 = nn.rename(columns={cases_prehelp[0]: all_tasks[0], cases_prehelp[1]: all_tasks[1],
                            cases_prehelp[2]: all_tasks[2]})
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
        au.plot_significant_dunn(data=n1n, dunn_result=dunn_result, fig_ax=ax, ly=1,
                                 pairs=significant_pairs, groups=all_tasks, x='Task', y='value', y_annotation=0)


    ax.set_ylim(-5, 25)
    ax.set_xticklabels([ 'Task 1 \n (CE: None)',
                        'Task 2 \n (CE: Task 1)', 'Task 3\n (CE: Tasks 1 & 2)'])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels([0, 5, 10, 15, 20])
    ax.set_ylabel('Expected Helpfulness')
    ax.set_xlabel('Tasks (Chronological order)')
    plt.text(-0.4, -3, 'CE: Collaboration Experience')
    plt.tight_layout()
    plt.savefig('result_helpfulness_task.eps', format='eps')
    plt.show()
