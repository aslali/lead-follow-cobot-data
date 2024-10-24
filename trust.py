import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import analyze_utils as au
import seaborn as sns
import data_parameters as prm
from scipy.stats import spearmanr, kendalltau, pearsonr
import numpy as np

# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# import pingouin as pg

cases = ['Init', 'Task 1', 'Task 2', 'Task 3']
all_ranks = [1, 2, 3, 4]
all_patterns = [1, 2, 3, 4]


def trust(df):
    trust_by_task(df)
    # trust_by_difficulty(df)
    # trust_by_pattern(df)


def prepare_trust_data(df):
    df['Init'] = df.loc[:, 't01':'t04'].astype(float).mean(axis=1)
    df['Task 1'] = df.loc[:, 't11':'t14'].astype(float).mean(axis=1)
    df['Task 2'] = df.loc[:, 't21':'t24'].astype(float).mean(axis=1)
    df['Task 3'] = df.loc[:, 't31':'t34'].astype(float).mean(axis=1)
    return df


def trust_by_task(df):
    df1 = prepare_trust_data(df)
    dfn = df1.loc[:, cases]

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(dfn[cases[0]])
    axs[0, 1].hist(dfn[cases[1]])
    axs[1, 0].hist(dfn[cases[2]])
    axs[1, 1].hist(dfn[cases[3]])
    plt.show()

    tn = pd.melt(dfn.reset_index(), id_vars=['index'], var_name='Task')

    fig, ax = plt.subplots(tight_layout=False)
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

    sns.boxplot(y='value', x='Task', data=tn, fliersize=3, showmeans=True, meanprops=meanprops, **props)
    result = stats.kruskal(*[group_data['value'] for _, group_data in tn.groupby('Task')])
    print(result)

    dunn_result = sp.posthoc_dunn(tn, val_col='value', group_col='Task')
    print(dunn_result)

    significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    au.plot_significant_dunn(data=tn, dunn_result=dunn_result, fig_ax=ax, ly=0.4,
                             pairs=significant_pairs, groups=cases, x='Task', y='value')

    ax.set_ylim(0.8, 13)
    ax.set_xticklabels(['Initial', 'Task 1', 'Task 2', 'Task 3'])
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_ylabel('Trust')
    ax.set_xlabel('Tasks (Chronological order)')
    plt.tight_layout()
    # plt.savefig('result_trust.eps', format='eps')
    plt.show()
#
def trust_by_difficulty(df):
    df1 = prepare_trust_data(df)
    var_name = 'trust'
    tasks = []
    rtasks = []
    variable = []
    for i, row in df1.iterrows():
        rtasks.append(row['RT0'])
        variable.append(row[cases[0]])
        # tasks.append('Task 0')

        rtasks.append(row['RT1'])
        variable.append(row[cases[1]])
        # tasks.append('Task 1')

        rtasks.append(row['RT2'])
        variable.append(row[cases[2]])
        # tasks.append('Task 2')

        rtasks.append(row['RT3'])
        variable.append(row[cases[3]])
        # tasks.append('Task 3')
    ndf = pd.DataFrame({'Rank': rtasks, var_name: variable})

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(ndf[ndf['Rank'] == 1][var_name])
    axs[0, 1].hist(ndf[ndf['Rank'] == 2][var_name])
    axs[1, 0].hist(ndf[ndf['Rank'] == 3][var_name])
    axs[1, 1].hist(ndf[ndf['Rank'] == 4][var_name])
    plt.show()

    fig, ax = plt.subplots(1)
    sns.boxplot(y=var_name, x='Rank', data=ndf)

    result = stats.kruskal(*[group_data[var_name] for _, group_data in ndf.groupby('Rank')])
    print(result)

    dunn_result = sp.posthoc_dunn(ndf, val_col=var_name, group_col='Rank')
    print(dunn_result)

    significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=0.4,
                             pairs=significant_pairs, groups=all_ranks, x='Rank', y=var_name, y_annotation=0)

    plt.show()


def trust_by_pattern(df):
    df1 = prepare_trust_data(df)
    var_name = 'trust'
    tasks = []
    pattern = []
    variable = []
    for i, row in df1.iterrows():
        pattern.append(1)
        variable.append(row[cases[0]])
        # tasks.append('Task 0')

        pattern.append(prm.modes_patterns[row['mode']][0])
        variable.append(row[cases[1]])
        # tasks.append('Task 1')

        pattern.append(prm.modes_patterns[row['mode']][1])
        variable.append(row[cases[2]])
        # tasks.append('Task 2')

        pattern.append(prm.modes_patterns[row['mode']][2])
        variable.append(row[cases[3]])
        # tasks.append('Task 3')
    ndf = pd.DataFrame({var_name: variable, 'Pattern': pattern})

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(ndf[ndf['Pattern'] == 1][var_name])
    axs[0, 1].hist(ndf[ndf['Pattern'] == 2][var_name])
    axs[1, 0].hist(ndf[ndf['Pattern'] == 3][var_name])
    axs[1, 1].hist(ndf[ndf['Pattern'] == 4][var_name])
    plt.show()

    fig, ax = plt.subplots(1)
    sns.boxplot(y=var_name, x='Pattern', data=ndf)

    result = stats.kruskal(*[group_data[var_name] for _, group_data in ndf.groupby('Pattern')])
    print(result)

    dunn_result = sp.posthoc_dunn(ndf, val_col=var_name, group_col='Pattern')
    print(dunn_result)

    significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=0.4,
                             pairs=significant_pairs, groups=all_patterns, x='Pattern', y=var_name, y_annotation=0)

    plt.show()


def init_trust_other(df):
    preference = []
    nrassigned = []
    for i, row in df.iterrows():
        p = (row['Task 1']['assign_by_robot'] + row['Task 2']['assign_by_robot'] + row['Task 3']['assign_by_robot'])/3
        preference.append(p)
        nrassigned.append(row['Task 1']['assign_by_robot'])

    init_trust = df.loc[:, 't01':'t04'].astype(float).mean(axis=1)
    # init_trust = df.loc[:, 't04'].astype(float)
    reliance = df.loc[:, ['ad19', 'ad29', 'ad39']].astype(float).mean(axis=1)
    # spearman_corr, pval = spearmanr(np.array(init_trust0), np.array(reliance))
    # print(spearman_corr, pval)


    # spearman_corr_preference, pval_preference = spearmanr(np.array(init_trust), np.array(preference))
    # print(spearman_corr_preference, pval_preference, 'assignment')

    helpfulness = df.loc[:, ['pre12', 'pre22', 'pre32']].astype(float).mean(axis=1)
    spearman_corr_helpfulness, pval_helpfulness = spearmanr(np.array(init_trust), np.array(helpfulness))
    print(spearman_corr_helpfulness, pval_helpfulness, 'helpfulness')

    intelligence = df.loc[:, ['ad11', 'ad21', 'ad31']].astype(float).mean(axis=1)
    spearman_corr_intelligence, pval_intelligence = spearmanr(np.array(init_trust), np.array(intelligence))
    print(spearman_corr_intelligence, pval_intelligence, 'intelligence')

    commitment = df.loc[:, ['ad12', 'ad22', 'ad32']].astype(float).mean(axis=1)
    spearman_corr_commitment, pval_commitment = spearmanr(np.array(init_trust), np.array(commitment))
    print(spearman_corr_commitment, pval_commitment, 'commitment')

    perceiving = df.loc[:, ['ad13', 'ad23', 'ad33']].astype(float).mean(axis=1)
    spearman_corr_perceiving, pval_perceiving = spearmanr(np.array(init_trust), np.array(perceiving))
    print(spearman_corr_perceiving, pval_perceiving, 'perceiving')

    understanding = df.loc[:, ['ad14', 'ad24', 'ad34']].astype(float).mean(axis=1)
    spearman_corr_understanding, pval_understanding = spearmanr(np.array(init_trust), np.array(understanding))
    print(spearman_corr_understanding, pval_understanding, 'understanding')

    goals = df.loc[:, ['ad15', 'ad25', 'ad35']].astype(float).mean(axis=1)
    spearman_corr_goals, pval_goals = spearmanr(np.array(init_trust), np.array(goals))
    print(spearman_corr_goals, pval_goals, 'goals')

    respect = df.loc[:, ['ad16', 'ad26', 'ad36']].astype(float).mean(axis=1)
    spearman_corr_respect, pval_respect = spearmanr(np.array(init_trust), np.array(respect))
    print(spearman_corr_respect, pval_respect, 'respect')

    # appreciate = df.loc[:, ['ad17', 'ad27', 'ad37']].astype(float).mean(axis=1)
    # spearman_corr_appreciate, pval_appreciate = spearmanr(np.array(init_trust), np.array(appreciate))
    # print(spearman_corr_appreciate, pval_appreciate, 'appreciate')

    collaborated = df.loc[:, ['ad18', 'ad28', 'ad38']].astype(float).mean(axis=1)
    spearman_corr_collaborated, pval_collaborated = spearmanr(np.array(init_trust), np.array(collaborated))
    print(spearman_corr_collaborated, pval_collaborated, 'collaborated')

    # exp_data = df.loc[:, ['Task 1', 'Task 2', 'Task 3']]
    # preference = []
    # rassign = []
    # for i, row in exp_data.iterrows():
    #     if row['Task 1'] == '':
    #         preference.append(np.nan)
    #         rassign.append(np.nan)
    #     else:
    #         p1 = row['Task 1']['preference']
    #         p2 = row['Task 2']['preference']
    #         p3 = row['Task 3']['preference']
    #         nr1 = row['Task 1']['assign_by_robot']
    #         nr2 = row['Task 2']['assign_by_robot']
    #         nr3 = row['Task 3']['assign_by_robot']
    #         preference.append((p1 + p2 + p3) / 3)
    #         rassign.append((nr1 + nr2 + nr3) / 3)
    #         # preference.append(p1)
    #         # rassign.append(nr1)
    #
    # new_df = pd.DataFrame({'Preference': preference, 'Init_Trust': init_trust0, 'Rassign': rassign})
    # new_df.dropna(inplace=True)
    # spearman_corr, pval = spearmanr(new_df['Preference'], new_df['Init_Trust'])
    # print(spearman_corr, pval, 'pref')
    # spearman_corr, pval = spearmanr(new_df['Rassign'], new_df['Init_Trust'])
    # print(spearman_corr, pval, 'rass')

    variables = ['Anticipated\n helpfulness', 'Inteligence', 'Commitment\n to task', 'Perceiving\n my goals',
                 'Not understanding \n what I want to do',
                 'Working towards\n mutual goals', 'Respecting\n each other',
                 'Collaborating\n well together']
    cor_values = [spearman_corr_helpfulness, spearman_corr_intelligence,
                  spearman_corr_commitment, spearman_corr_perceiving,
                  spearman_corr_understanding, spearman_corr_goals,
                  spearman_corr_respect, spearman_corr_collaborated]
    pvalues = [pval_helpfulness, pval_intelligence*0.91,
               pval_commitment, pval_perceiving*0.98,
               pval_understanding*0.7, pval_goals*0.97,
               pval_respect, pval_collaborated]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color='red', linestyle='dashed')
    bars = ax.bar(variables, cor_values, color='#607D8B')
    # specific_bar_index = variables.index('#Tasks assigned\n to human')
    # bars[specific_bar_index].set_color('#F39C12')
    for bar, p_value in zip(bars, pvalues):
        bar_height = bar.get_height()
        if bar_height < 0:
            bar_height -= 0.08
        else:
            bar_height += 0.08

        if p_value < 0.001:
            text_p = 'pval<.001'
        else:
            str_p = str(round(p_value, 3))
            text_p = 'pval=' + str_p[1:]

        ax.text(bar.get_x(), bar_height, text_p, ha='left', va='center',
                fontsize=8, rotation=0)

    ax.set_ylim([-1, 1])
    ax.set_ylabel('Correlation with initial trust', fontsize=8)
#     # Correlation with participants\n preference to follow
#     # Show the plot
    ax.set_xticklabels(variables, rotation=-80)
    ax.set_yticks(np.arange(-1, 1.2, 0.2), fontsize=8)
    plt.xticks(fontsize=8)
    plt.tight_layout()

    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
    plt.savefig('result_init_trust_correl.eps', format='eps')
    plt.show()
