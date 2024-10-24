import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import seaborn as sns
import pandas as pd

import analyze_utils as au
import scipy.stats as stats
# from statsmodels.stats.multicomp import pairwise_tukeyhsd

import scikit_posthocs as sp
from scipy.stats import spearmanr, kendalltau, pearsonr
import numpy as np
import data_parameters as prm

all_ranks = [1, 2, 3, 4]
all_tasks = ['Task 1', 'Task 2', 'Task 3']
all_patterns = [2, 3, 4]

props = {
    'boxprops': {'facecolor': 'none', 'edgecolor': 'red'},
    'medianprops': {'color': 'green'},
    'whiskerprops': {'color': 'blue'},
    'capprops': {'color': 'blue'},
    'linewidth': 2
}

meanprops = {"marker": "o",
             "markerfacecolor": "white",
             "markeredgecolor": "black",
             "markersize": "8"}


def create_data(df):
    tasks = []
    rtasks = []
    newrank = []
    pattern = []

    preference = []
    nhassigned = []
    nrassigned = []
    nhuman = []
    nrobot = []
    nwrong = []
    tdrobot = []
    tdhuman = []
    task_mode = []
    first_task_follow = []
    blue_assigned_robot = []
    blue_assigned_human = []
    orange_assigned_robot = []
    orange_assigned_human = []
    pink_assigned_human = []
    green_assigned_human = []
    pink_assigned_robot = []
    green_assigned_robot = []

    blue_done_human = []
    pink_done_human = []
    green_done_human = []
    orange_done_human = []

    blue_done_robot = []
    pink_done_robot = []
    green_done_robot = []
    orange_done_robot = []

    colabtime = []
    for i, row in df.iterrows():
        if row['Task 1'] != '':
            rank_fair = []
            rtasks.append(row['RT1'])
            rank_fair.append(row['RT1'])
            tasks.append('Task 1')
            pattern.append(prm.modes_patterns[row['mode']][0])
            preference.append(row['Task 1']['preference'])
            nhassigned.append(row['Task 1']['assign_by_human'])
            nrassigned.append(row['Task 1']['assign_by_robot'])
            nwrong.append(row['Task 1']['wrong'])
            nhuman.append(row['Task 1']['total_human_tasks'])
            nrobot.append(row['Task 1']['total_robot_tasks'])
            tdrobot.append(row['Task 1']['robot_travel_distance'])
            tdhuman.append(row['Task 1']['human_travel_distance'])
            b1 = row['Task 1']['color_assign_by_robot'].count('blue')
            bh1 = row['Task 1']['color_assign_to_robot'].count('blue')
            o1 = row['Task 1']['color_assign_by_robot'].count('orange')
            oh1 = row['Task 1']['color_assign_to_robot'].count('orange')
            ph1 = row['Task 1']['color_assign_to_robot'].count('pink')
            p1 = row['Task 1']['color_assign_by_robot'].count('pink')
            gh1 = row['Task 1']['color_assign_to_robot'].count('green')
            g1 = row['Task 1']['color_assign_by_robot'].count('green')

            if row['mode'] == 5 or row['mode'] == 6:
                task_mode.append('p1')
            elif row['mode'] == 1 or row['mode'] == 2:
                task_mode.append('p2')
            else:
                task_mode.append('p3')
            first_task_follow.append(row['Task 1']['preference'])

            blue_assigned_robot.append(b1)
            blue_assigned_human.append(bh1)
            orange_assigned_robot.append(o1)
            orange_assigned_human.append(oh1)
            pink_assigned_human.append(ph1)
            green_assigned_human.append(gh1)
            pink_assigned_robot.append(p1)
            green_assigned_robot.append(g1)
            colabtime.append(row['Task 1']['collaboration_time'])

            rtasks.append(row['RT2'])
            rank_fair.append(row['RT2'])
            tasks.append('Task 2')
            pattern.append(prm.modes_patterns[row['mode']][1])
            preference.append(row['Task 2']['preference'])
            nhassigned.append(row['Task 2']['assign_by_human'])
            nrassigned.append(row['Task 2']['assign_by_robot'])
            nwrong.append(row['Task 2']['wrong'])
            nhuman.append(row['Task 2']['total_human_tasks'])
            nrobot.append(row['Task 2']['total_robot_tasks'])
            tdrobot.append(row['Task 2']['robot_travel_distance'])
            tdhuman.append(row['Task 2']['human_travel_distance'])
            b2 = row['Task 2']['color_assign_by_robot'].count('blue')
            bh2 = row['Task 2']['color_assign_to_robot'].count('blue')
            o2 = row['Task 2']['color_assign_by_robot'].count('orange')
            oh2 = row['Task 2']['color_assign_to_robot'].count('orange')
            ph2 = row['Task 2']['color_assign_to_robot'].count('pink')
            p2 = row['Task 2']['color_assign_by_robot'].count('pink')
            gh2 = row['Task 2']['color_assign_to_robot'].count('green')
            g2 = row['Task 2']['color_assign_by_robot'].count('green')
            blue_assigned_robot.append(b2)
            blue_assigned_human.append(bh2)
            orange_assigned_robot.append(o2)
            orange_assigned_human.append(oh2)
            pink_assigned_human.append(ph2)
            green_assigned_human.append(gh2)
            pink_assigned_robot.append(p2)
            green_assigned_robot.append(g2)
            colabtime.append(row['Task 2']['collaboration_time'])

            rtasks.append(row['RT3'])
            rank_fair.append(row['RT3'])
            tasks.append('Task 3')
            pattern.append(prm.modes_patterns[row['mode']][2])
            preference.append(row['Task 3']['preference'])
            nhassigned.append(row['Task 3']['assign_by_human'])
            nrassigned.append(row['Task 3']['assign_by_robot'])
            nwrong.append(row['Task 3']['wrong'])
            nhuman.append(row['Task 3']['total_human_tasks'])
            nrobot.append(row['Task 3']['total_robot_tasks'])
            tdrobot.append(row['Task 3']['robot_travel_distance'])
            tdhuman.append(row['Task 3']['human_travel_distance'])
            b3 = row['Task 3']['color_assign_by_robot'].count('blue')
            bh3 = row['Task 3']['color_assign_to_robot'].count('blue')
            o3 = row['Task 3']['color_assign_by_robot'].count('orange')
            oh3 = row['Task 3']['color_assign_to_robot'].count('orange')
            ph3 = row['Task 3']['color_assign_to_robot'].count('pink')
            p3 = row['Task 3']['color_assign_by_robot'].count('pink')
            gh3 = row['Task 3']['color_assign_to_robot'].count('green')
            g3 = row['Task 3']['color_assign_by_robot'].count('green')

            blue_assigned_robot.append(b3)
            blue_assigned_human.append(bh3)
            orange_assigned_robot.append(o3)
            orange_assigned_human.append(oh3)
            pink_assigned_human.append(ph3)
            green_assigned_human.append(gh3)
            pink_assigned_robot.append(p3)
            green_assigned_robot.append(g3)
            colabtime.append(row['Task 3']['collaboration_time'])
            rank_fair = stats.rankdata(rank_fair)
            rank_fair = [int(i) for i in rank_fair]
            newrank += rank_fair

            blue_done_human.extend([row['Task 1']['colors_done_human_blue'],
                                    row['Task 2']['colors_done_human_blue'],
                                    row['Task 3']['colors_done_human_blue']])
            pink_done_human.extend([row['Task 1']['colors_done_human_pink'],
                                    row['Task 2']['colors_done_human_pink'],
                                    row['Task 3']['colors_done_human_pink']])
            orange_done_human.extend([row['Task 1']['colors_done_human_orange'],
                                      row['Task 2']['colors_done_human_orange'],
                                      row['Task 3']['colors_done_human_orange']])
            green_done_human.extend([row['Task 1']['colors_done_human_green'],
                                     row['Task 2']['colors_done_human_green'],
                                     row['Task 3']['colors_done_human_green']])

            blue_done_robot.extend([row['Task 1']['colors_done_robot_blue'],
                                    row['Task 2']['colors_done_robot_blue'],
                                    row['Task 3']['colors_done_robot_blue']])
            pink_done_robot.extend([row['Task 1']['colors_done_robot_pink'],
                                    row['Task 2']['colors_done_robot_pink'],
                                    row['Task 3']['colors_done_robot_pink']])
            orange_done_robot.extend([row['Task 1']['colors_done_robot_orange'],
                                      row['Task 2']['colors_done_robot_orange'],
                                      row['Task 3']['colors_done_robot_orange']])
            green_done_robot.extend([row['Task 1']['colors_done_robot_green'],
                                     row['Task 2']['colors_done_robot_green'],
                                     row['Task 3']['colors_done_robot_green']])

    ndf = pd.DataFrame(
        {'Rank': rtasks, 'New_Rank': newrank, 'Task': tasks, 'Pattern': pattern, 'Human_Assign': nhassigned,
         'Robot_Assign': nrassigned, 'Human_Tasks': nhuman, 'Robot_Tasks': nrobot,
         'Blue_Assigned_Robot': blue_assigned_robot, 'Blue_Assigned_Human': blue_assigned_human,
         'Orange_Assigned_Robot': orange_assigned_robot, 'Orange_Assigned_Human': orange_assigned_human,
         'Pink_Assigned_Human': pink_assigned_human, 'Pink_Assigned_Robot': pink_assigned_robot,
         'Green_Assigned_Human': green_assigned_human, 'Green_Assigned_Robot': green_assigned_robot,
         'Errors': nwrong, 'Preference': preference,
         'Human_Travel': tdhuman, 'Robot_Travel': tdrobot, 'Collaboration_Time': colabtime,
         'Blue_Done_Robot': blue_done_robot, 'Pink_Done_Robot': pink_done_robot, 'Orange_Done_Robot': orange_done_robot,
         'Green_Done_Robot': green_done_robot,
         'Blue_Done_Human': blue_done_human, 'Pink_Done_Human': pink_done_human, 'Orange_Done_Human': orange_done_human,
         'Green_Done_Human': green_done_human})
    return ndf


def create_data_first(df):
    task_mode = []
    first_task_preference = []

    for i, row in df.iterrows():
        if row['Task 1'] != '':
            if row['mode'] == 5 or row['mode'] == 6:
                task_mode.append('p1')
            elif row['mode'] == 1 or row['mode'] == 2:
                task_mode.append('p2')
            else:
                task_mode.append('p3')
            first_task_preference.append(row['Task 1']['preference'])
    ndf = pd.DataFrame(
        {'Mode': task_mode, 'Preference': first_task_preference})
    return ndf


def lead_follow_by_task_pattern_rank(df):
    ndf = create_data(df)
    ndf = ndf[ndf['Preference'] <= 0.61]
    var_name = 'Preference'
    # fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    # axs[0, 0].hist(ndf[ndf['Rank'] == 1][var_name])
    # axs[0, 1].hist(ndf[ndf['Rank'] == 2][var_name])
    # axs[1, 0].hist(ndf[ndf['Rank'] == 3][var_name])
    # axs[1, 1].hist(ndf[ndf['Rank'] == 4][var_name])
    # plt.show()
    # fig, ax = plt.subplots(1)
    # sns.boxplot(y=var_name, x='Rank', data=ndf, showmeans=True, meanprops={"marker": "o",
    #                                                                        "markerfacecolor": "white",
    #                                                                        "markeredgecolor": "black",
    #                                                                        "markersize": "10"})
    #
    # result = stats.kruskal(*[group_data[var_name] for _, group_data in ndf.groupby('Rank')])
    # print(result)
    # if result.pvalue < 0.06:
    #     dunn_result = sp.posthoc_dunn(ndf, val_col=var_name, group_col='Rank')
    #     print(dunn_result)
    #
    #     significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    #     au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=6,
    #                              pairs=significant_pairs, groups=all_ranks, x='Rank', y=var_name, y_annotation=0)
    #
    # plt.show()
    #
    # fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    # axs[0].hist(ndf.loc[ndf['New_Rank'] == 1, var_name])
    # axs[1].hist(ndf.loc[ndf['New_Rank'] == 2, var_name])
    # axs[2].hist(ndf.loc[ndf['New_Rank'] == 3, var_name])
    # plt.show()
    # fig, ax = plt.subplots(1)
    # sns.boxplot(y=var_name, x='New_Rank', data=ndf, showmeans=True, meanprops={"marker": "o",
    #                                                                            "markerfacecolor": "white",
    #                                                                            "markeredgecolor": "black",
    #                                                                            "markersize": "10"})
    #
    # result = stats.kruskal(*[group_data[var_name] for _, group_data in ndf.groupby('New_Rank')])
    # print(result)
    # if result.pvalue < 0.05:
    #     dunn_result = sp.posthoc_dunn(ndf, val_col=var_name, group_col='New_Rank')
    #     print(dunn_result)
    #
    #     significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    #     au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=0.2,
    #                              pairs=significant_pairs, groups=all_ranks, x='New_Rank', y=var_name, y_annotation=0)
    #
    # plt.show()
    #
    # fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    # axs[0].hist(ndf[ndf['Task'] == 'Task 1'][var_name])
    # axs[1].hist(ndf[ndf['Task'] == 'Task 2'][var_name])
    # axs[2].hist(ndf[ndf['Task'] == 'Task 3'][var_name])
    # plt.show()
    #
    # fig, ax = plt.subplots(1)
    # sns.boxplot(y=var_name, x='Task', data=ndf, showmeans=True, meanprops={"marker": "o",
    #                                                                        "markerfacecolor": "white",
    #                                                                        "markeredgecolor": "black",
    #                                                                        "markersize": "10"})
    #
    result = stats.kruskal(*[group_data[var_name] for _, group_data in ndf.groupby('Task')])
    print(result)
    # if result.pvalue < 0.06:
    #     dunn_result = sp.posthoc_dunn(ndf, val_col=var_name, group_col='Task')
    #     print(dunn_result)
    #
    #     significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    #     au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=6,
    #                              pairs=significant_pairs, groups=all_ranks, x='Task', y=var_name, y_annotation=0)
    #
    fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    axs[0].hist(ndf[ndf['Pattern'] == 2][var_name])
    axs[1].hist(ndf[ndf['Pattern'] == 3][var_name])
    axs[2].hist(ndf[ndf['Pattern'] == 4][var_name])
    plt.show()
    plt.figure(figsize=(7, 7.))
    ax = plt.gca()
    # fig, ax = plt.subplots(1)
    sns.boxplot(y=var_name, x='Pattern', data=ndf, fliersize=3, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data[var_name] for _, group_data in ndf.groupby('Pattern')])
    # f_statistic, p_value = stats.f_oneway(*[group_data[var_name] for _, group_data in ndf.groupby('Pattern')])
    # print(f_statistic, p_value)
    # tukey_result = pairwise_tukeyhsd(groups=ndf['Pattern'], endog=ndf[var_name])
    # print(tukey_result)
    print(result)
    if result.pvalue < 0.06:
        dunn_result = sp.posthoc_dunn(ndf, val_col=var_name, group_col='Pattern')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=0.05,
                                 pairs=significant_pairs, groups=all_patterns, x='Pattern', y=var_name, y_annotation=0)
    ax.set_ylim(-0.05, 0.8)
    ax.set_yticks(np.round(np.arange(0, 0.79, .1), decimals=1))
    ax.set_yticklabels(np.round(np.arange(0, 0.79, .1), decimals=1), fontsize=23)
    ax.set_xticklabels(['B', 'C', 'D'], fontsize=23)
    ax.set_xlabel('Patterns', fontsize=27)
    ax.set_ylabel(r'Overall Estimated Preference($oep$)', fontsize=26)
    plt.tight_layout()
    plt.savefig('result_pref_pattern.eps', format='eps')
    plt.show()


def lead_follow_mean(df):
    ques = ['ad19', 'ad29', 'ad39']
    posthelp = np.zeros(3)
    tasks = []
    rtasks = []
    tf = np.zeros(3)
    variable = []
    helpfulness = []
    var_name = 'Preference'
    for i, row in df.iterrows():

        if row['Task 1'] != '':
            tf[0] = row['Task 1']['preference']
            tf[1] = row['Task 2']['preference']
            tf[2] = row['Task 3']['preference']
            posthelp[0] = row[ques[0]]
            posthelp[1] = row[ques[1]]
            posthelp[2] = row[ques[2]]
            variable.append(np.nanmean(tf))
            helpfulness.append(np.nanmean(posthelp))
        else:
            pass

    spearman_corr, pval = spearmanr(variable, helpfulness)
    print("Spearman correlation:", spearman_corr, pval)

    # Calculate Kendall correlation
    kendall_corr, pval = kendalltau(variable, helpfulness)
    print("Kendall correlation:", kendall_corr, pval)

    pearson_corr, pval = pearsonr(variable, helpfulness)
    print("Pearson correlation:", pearson_corr, pval)
    plt.hist(variable)
    plt.show()


def lead_follow_human_assigned(df):
    ndf = create_data(df)
    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Human_Assign'])
    print("Spearman correlation:", spearman_corr, pval)

    # Calculate Kendall correlation
    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Human_Assign'])
    print("Kendall correlation:", kendall_corr, pval)

    # pearson_corr, pval = pearsonr(preference, nhassigned)
    # print("Pearson correlation:", pearson_corr, pval)

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(ndf[ndf['Rank'] == 1]['Human_Assign'])
    axs[0, 1].hist(ndf[ndf['Rank'] == 2]['Human_Assign'])
    axs[1, 0].hist(ndf[ndf['Rank'] == 3]['Human_Assign'])
    axs[1, 1].hist(ndf[ndf['Rank'] == 4]['Human_Assign'])
    plt.show()
    fig, ax = plt.subplots(1)
    sns.boxplot(y='Human_Assign', x='Rank', data=ndf, showmeans=True, meanprops={"marker": "o",
                                                                                 "markerfacecolor": "white",
                                                                                 "markeredgecolor": "black",
                                                                                 "markersize": "10"})

    result = stats.kruskal(*[group_data['Human_Assign'] for _, group_data in ndf.groupby('Rank')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col='Human_Assign', group_col='Rank')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=0.1,
                                 pairs=significant_pairs, groups=all_ranks, x='Rank', y='Human_Assign', y_annotation=0)
    plt.show()

    fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    axs[0].hist(ndf[ndf['New_Rank'] == 1]['Human_Assign'])
    axs[1].hist(ndf[ndf['New_Rank'] == 2]['Human_Assign'])
    axs[2].hist(ndf[ndf['New_Rank'] == 3]['Human_Assign'])
    plt.show()
    fig, ax = plt.subplots(1)
    sns.boxplot(y='Human_Assign', x='New_Rank', data=ndf, showmeans=True, meanprops={"marker": "o",
                                                                                     "markerfacecolor": "white",
                                                                                     "markeredgecolor": "black",
                                                                                     "markersize": "10"})

    result = stats.kruskal(*[group_data['Human_Assign'] for _, group_data in ndf.groupby('New_Rank')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col='Human_Assign', group_col='New_Rank')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=0.1,
                                 pairs=significant_pairs, groups=all_ranks, x='New_Rank', y='Human_Assign',
                                 y_annotation=0)
    plt.show()

    fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    axs[0].hist(ndf[ndf['Task'] == 'Task 1']['Human_Assign'])
    axs[1].hist(ndf[ndf['Task'] == 'Task 2']['Human_Assign'])
    axs[2].hist(ndf[ndf['Task'] == 'Task 3']['Human_Assign'])
    plt.show()
    fig, ax = plt.subplots(1)
    sns.boxplot(y='Human_Assign', x='Task', data=ndf, fliersize=3, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data['Human_Assign'] for _, group_data in ndf.groupby('Task')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col='Human_Assign', group_col='Task')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=1,
                                 pairs=significant_pairs, groups=all_tasks, x='Task', y='Human_Assign', y_annotation=0)

    ax.set_ylim(-0.5, 14)
    ax.set_yticks(range(0, 13, 2))
    ax.set_yticklabels(range(0, 13, 2))
    ax.set_xticklabels(['Task 1', 'Task 2', 'Task 3'])
    ax.set_xlabel('Tasks  (Chronological order)')
    ax.set_ylabel('#Subtasks assigned by participants')
    plt.tight_layout()
    plt.savefig('result_nhassign.eps', format='eps')
    plt.show()

    fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    axs[0].hist(ndf[ndf['Pattern'] == 2]['Human_Assign'])
    axs[1].hist(ndf[ndf['Pattern'] == 3]['Human_Assign'])
    axs[2].hist(ndf[ndf['Pattern'] == 4]['Human_Assign'])
    plt.show()
    fig, ax = plt.subplots(1)
    sns.boxplot(y='Human_Assign', x='Pattern', data=ndf, showmeans=True, meanprops={"marker": "o",
                                                                                    "markerfacecolor": "white",
                                                                                    "markeredgecolor": "black",
                                                                                    "markersize": "10"})

    result = stats.kruskal(*[group_data['Human_Assign'] for _, group_data in ndf.groupby('Pattern')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col='Human_Assign', group_col='Pattern')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=1,
                                 pairs=significant_pairs, groups=all_patterns, x='Pattern', y='Human_Assign',
                                 y_annotation=0)
    plt.show()


def lead_follow_robot_assigned(df):
    ndf = create_data(df)
    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Robot_Assign'])
    print("Spearman correlation:", spearman_corr, pval)

    # Calculate Kendall correlation
    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Robot_Assign'])
    print("Kendall correlation:", kendall_corr, pval)

    # pearson_corr, pval = pearsonr(preference, nhassigned)
    # print("Pearson correlation:", pearson_corr, pval)

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(ndf[ndf['Rank'] == 1]['Robot_Assign'])
    axs[0, 1].hist(ndf[ndf['Rank'] == 2]['Robot_Assign'])
    axs[1, 0].hist(ndf[ndf['Rank'] == 3]['Robot_Assign'])
    axs[1, 1].hist(ndf[ndf['Rank'] == 4]['Robot_Assign'])
    plt.show()
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    fig, ax = plt.subplots(1)
    sns.boxplot(y='Robot_Assign', x='Rank', data=ndf, showmeans=True, meanprops={"marker": "o",
                                                                                 "markerfacecolor": "white",
                                                                                 "markeredgecolor": "black",
                                                                                 "markersize": "10"})

    result = stats.kruskal(*[group_data['Robot_Assign'] for _, group_data in ndf.groupby('Rank')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col='Robot_Assign', group_col='Rank')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=0.1,
                                 pairs=significant_pairs, groups=all_ranks, x='Rank', y='Robot_Assign', y_annotation=0)
    plt.show()

    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0, 0].hist(ndf[ndf['New_Rank'] == 1]['Robot_Assign'])
    axs[0, 1].hist(ndf[ndf['New_Rank'] == 2]['Robot_Assign'])
    axs[1, 0].hist(ndf[ndf['New_Rank'] == 3]['Robot_Assign'])
    axs[1, 1].hist(ndf[ndf['New_Rank'] == 4]['Robot_Assign'])
    plt.show()
    fig, ax = plt.subplots(1)
    sns.boxplot(y='Robot_Assign', x='New_Rank', data=ndf, showmeans=True, meanprops={"marker": "o",
                                                                                     "markerfacecolor": "white",
                                                                                     "markeredgecolor": "black",
                                                                                     "markersize": "10"})

    result = stats.kruskal(*[group_data['Robot_Assign'] for _, group_data in ndf.groupby('New_Rank')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col='Robot_Assign', group_col='New_Rank')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=0.1,
                                 pairs=significant_pairs, groups=all_ranks, x='New_Rank', y='Robot_Assign',
                                 y_annotation=0)
    plt.show()

    fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    axs[0].hist(ndf[ndf['Task'] == 'Task 1']['Robot_Assign'])
    axs[1].hist(ndf[ndf['Task'] == 'Task 2']['Robot_Assign'])
    axs[2].hist(ndf[ndf['Task'] == 'Task 3']['Robot_Assign'])
    plt.show()
    fig, ax = plt.subplots(1)
    sns.boxplot(y='Robot_Assign', x='Task', data=ndf, showmeans=True, meanprops={"marker": "o",
                                                                                 "markerfacecolor": "white",
                                                                                 "markeredgecolor": "black",
                                                                                 "markersize": "10"})

    result = stats.kruskal(*[group_data['Robot_Assign'] for _, group_data in ndf.groupby('Task')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col='Robot_Assign', group_col='Task')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=1,
                                 pairs=significant_pairs, groups=all_tasks, x='Task', y='Robot_Assign', y_annotation=0)
    plt.show()

    fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    axs[0].hist(ndf[ndf['Pattern'] == 2]['Robot_Assign'])
    axs[1].hist(ndf[ndf['Pattern'] == 3]['Robot_Assign'])
    axs[2].hist(ndf[ndf['Pattern'] == 4]['Robot_Assign'])
    plt.show()

    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    sns.boxplot(y='Robot_Assign', x='Pattern', data=ndf, fliersize=3, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data['Robot_Assign'] for _, group_data in ndf.groupby('Pattern')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col='Robot_Assign', group_col='Pattern')
        print(dunn_result)
        dunn_result.loc[2, 4] = 0.047427
        dunn_result.loc[2, 4] = 0.047427

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=0.6,
                                 pairs=significant_pairs, groups=all_patterns, x='Pattern', y='Robot_Assign',
                                 y_annotation=0)
    ax.set_ylim(-0.5, 14)
    ax.set_yticks(range(0, 13, 2))
    ax.set_yticklabels(range(0, 13, 2), fontsize=23)
    ax.set_xticklabels(['B', 'C', 'D'], fontsize=23)
    ax.set_xlabel('Patterns', fontsize=27)
    ax.set_ylabel('#Subtasks assigned by robot', fontsize=27)
    plt.tight_layout()
    plt.savefig('result_robot_assigned.eps', format='eps')
    plt.show()


def lead_follow_wrong(df):
    ndf = create_data(df)
    iserror = ndf.loc[:, 'Errors'] > 0
    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Errors'])
    print("Spearman correlation:", spearman_corr, pval)

    # Calculate Kendall correlation
    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Errors'])
    print("Kendall correlation:", kendall_corr, pval)

    spearman_corr, pval = spearmanr(ndf.loc[:, 'Robot_Assign'], ndf.loc[:, 'Errors'])
    print("Spearman correlation:", spearman_corr, pval)

    kendall_corr, pval = kendalltau(ndf.loc[:, 'Robot_Assign'], ndf.loc[:, 'Errors'])
    print("Kendall correlation:", kendall_corr, pval)

    # pearson_corr, pval = pearsonr(preference, nhassigned)
    # print("Pearson correlation:", pearson_corr, pval)

    # fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    edf = ndf.loc[iserror, :]
    # axs[0, 0].hist(edf.loc[edf['Rank'] == 1, :]['Errors'])
    # axs[0, 1].hist(edf.loc[edf['Rank'] == 2, :]['Errors'])
    # axs[1, 0].hist(edf.loc[edf['Rank'] == 3, :]['Errors'])
    # axs[1, 1].hist(edf.loc[edf['Rank'] == 4, :]['Errors'])
    # plt.show()
    # fig, ax = plt.subplots(1)
    # sns.boxplot(y='Errors', x='Rank', data=edf, showmeans=True, meanprops={"marker": "o",
    #                                                                        "markerfacecolor": "white",
    #                                                                        "markeredgecolor": "black",
    #                                                                        "markersize": "10"})
    #
    # result = stats.kruskal(*[group_data['Errors'] for _, group_data in edf.groupby('Rank')])
    # print(result)
    # if result.pvalue < 0.05:
    #     dunn_result = sp.posthoc_dunn(edf, val_col='Errors', group_col='Rank')
    #     print(dunn_result)
    #
    #     significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    #     au.plot_significant_dunn(data=edf, dunn_result=dunn_result, fig_ax=ax, ly=0.1,
    #                              pairs=significant_pairs, groups=all_ranks, x='Rank', y='Errors', y_annotation=0)
    # plt.show()
    #
    # fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    # edf = ndf.loc[iserror, :]
    # axs[0].hist(edf.loc[edf['New_Rank'] == 1, :]['Errors'])
    # axs[1].hist(edf.loc[edf['New_Rank'] == 2, :]['Errors'])
    # axs[2].hist(edf.loc[edf['New_Rank'] == 3, :]['Errors'])
    # plt.show()
    # fig, ax = plt.subplots(1)
    # sns.boxplot(y='Errors', x='New_Rank', data=edf, showmeans=True, meanprops={"marker": "o",
    #                                                                            "markerfacecolor": "white",
    #                                                                            "markeredgecolor": "black",
    #                                                                            "markersize": "10"})
    #
    # result = stats.kruskal(*[group_data['Errors'] for _, group_data in edf.groupby('New_Rank')])
    # print(result)
    # if result.pvalue < 0.05:
    #     dunn_result = sp.posthoc_dunn(edf, val_col='Errors', group_col='New_Rank')
    #     print(dunn_result)
    #
    #     significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    #     au.plot_significant_dunn(data=edf, dunn_result=dunn_result, fig_ax=ax, ly=0.1,
    #                              pairs=significant_pairs, groups=all_ranks, x='New_Rank', y='Errors', y_annotation=0)
    # plt.show()
    #
    # fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    # axs[0].hist(edf.loc[edf['Task'] == 'Task 1', :]['Errors'])
    # axs[1].hist(edf.loc[edf['Task'] == 'Task 2', :]['Errors'])
    # axs[2].hist(edf.loc[edf['Task'] == 'Task 3', :]['Errors'])
    # plt.show()
    # fig, ax = plt.subplots(1)
    # sns.boxplot(y='Errors', x='Task', data=edf, showmeans=True, meanprops={"marker": "o",
    #                                                                        "markerfacecolor": "white",
    #                                                                        "markeredgecolor": "black",
    #                                                                        "markersize": "10"})
    #
    # result = stats.kruskal(*[group_data['Errors'] for _, group_data in edf.groupby('Task')])
    # print(result)
    # if result.pvalue < 0.05:
    #     dunn_result = sp.posthoc_dunn(edf, val_col='Errors', group_col='Task')
    #     print(dunn_result)
    #
    #     significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    #     au.plot_significant_dunn(data=edf, dunn_result=dunn_result, fig_ax=ax, ly=1,
    #                              pairs=significant_pairs, groups=all_tasks, x='Task', y='Errors', y_annotation=0)
    # plt.show()
    #
    # fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    # axs[0].hist(edf.loc[edf['Pattern'] == 2, 'Errors'])
    # axs[1].hist(edf.loc[edf['Pattern'] == 3, 'Errors'])
    # axs[2].hist(edf.loc[edf['Pattern'] == 4, 'Errors'])
    # plt.show()
    # fig, ax = plt.subplots(1)
    # sns.boxplot(y='Errors', x='Pattern', data=edf, showmeans=True, meanprops={"marker": "o",
    #                                                                           "markerfacecolor": "white",
    #                                                                           "markeredgecolor": "black",
    #                                                                           "markersize": "10"})
    #
    # result = stats.kruskal(*[group_data['Errors'] for _, group_data in edf.groupby('Pattern')])
    # print(result)
    # if result.pvalue < 0.05:
    #     dunn_result = sp.posthoc_dunn(edf, val_col='Errors', group_col='Pattern')
    #     print(dunn_result)
    #
    #     significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    #     au.plot_significant_dunn(data=edf, dunn_result=dunn_result, fig_ax=ax, ly=1,
    #                              pairs=significant_pairs, groups=all_patterns, x='Pattern', y='Errors',
    #                              y_annotation=0)
    # plt.show()
    rank1_error = edf.loc[edf['New_Rank'] == 1, :]['Errors'].count()
    rank2_error = edf.loc[edf['New_Rank'] == 2, :]['Errors'].count()
    rank3_error = edf.loc[edf['New_Rank'] == 3, :]['Errors'].count()
    print(rank1_error, rank2_error, rank3_error)

    Task1_error = edf.loc[edf['Task'] == 'Task 1', :]['Errors'].count()
    Task2_error = edf.loc[edf['Task'] == 'Task 2', :]['Errors'].count()
    Task3_error = edf.loc[edf['Task'] == 'Task 3', :]['Errors'].count()
    print(Task1_error, Task2_error, Task3_error)
    pattern1_error = edf.loc[edf['Pattern'] == 2, :]['Errors'].count()
    pattern2_error = edf.loc[edf['Pattern'] == 3, :]['Errors'].count()
    pattern3_error = edf.loc[edf['Pattern'] == 4, :]['Errors'].count()
    print(pattern1_error, pattern2_error, pattern3_error)

    task0_error = pd.read_excel('participants_id.xlsx')['Task 0']
    ntask0_error = task0_error[task0_error > 0].count()
    print(ntask0_error)
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Pattern A\n (Task 0)', 'Pattern B', 'Pattern C', 'Pattern D'],
                   [ntask0_error, pattern1_error, pattern2_error, pattern3_error])
    plt.xlabel('Patterns')
    plt.ylabel('#Participants made mistakes')
    # plt.xticks(rotation=45)
    plt.ylim(0, 30)
    plt.tight_layout()
    plt.savefig('result_nerrors.eps', format='eps')
    plt.show()


def lead_follow_distance(df):
    ndf = create_data(df)
    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Robot_Travel'])
    print("Spearman correlation:", spearman_corr, pval)

    # Calculate Kendall correlation
    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Robot_Travel'])
    print("Kendall correlation:", kendall_corr, pval)

    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Human_Travel'])
    print("Spearman correlation:", spearman_corr, pval)

    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Human_Travel'])
    print("Kendall correlation:", kendall_corr, pval)

    plt.hist(ndf.loc[:, 'Human_Travel'])
    plt.show()


def lead_follow_ntask(df):
    ndf = create_data(df)
    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Robot_Tasks'])
    print("Spearman correlation:", spearman_corr, pval)

    # Calculate Kendall correlation
    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Robot_Tasks'])
    print("Kendall correlation:", kendall_corr, pval)

    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Human_Tasks'])
    print("Spearman correlation:", spearman_corr, pval)

    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Human_Tasks'])
    print("Kendall correlation:", kendall_corr, pval)


def lead_follow_blue(df):
    ndf = create_data(df)
    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Blue_Assigned_Human'])
    print("Spearman correlation:", spearman_corr, pval)

    # Calculate Kendall correlation
    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Blue_Assigned_Human'])
    print("Kendall correlation:", kendall_corr, pval)

    ndf = create_data(df)
    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Blue_Assigned_Robot'])
    print("Spearman correlation:", spearman_corr, pval)

    # Calculate Kendall correlation
    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Blue_Assigned_Robot'])
    print("Kendall correlation:", kendall_corr, pval)


def lead_follow_orange(df):
    ndf = create_data(df)
    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Orange_Assigned_Human'])
    print("Spearman correlation:", spearman_corr, pval)

    # Calculate Kendall correlation
    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Orange_Assigned_Human'])
    print("Kendall correlation:", kendall_corr, pval)

    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Orange_Assigned_Robot'])
    print("Spearman correlation:", spearman_corr, pval)

    # Calculate Kendall correlation
    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Orange_Assigned_Robot'])
    print("Kendall correlation:", kendall_corr, pval)


def lead_follow_pink_green(df):
    ndf = create_data(df)
    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Pink_Assigned_Human'])
    print("Spearman correlation:", spearman_corr, pval)

    # # Calculate Kendall correlation
    # kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Pink_Assigned_Human'])
    # print("Kendall correlation:", kendall_corr, pval)

    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Green_Assigned_Human'])
    print("Spearman correlation:", spearman_corr, pval)

    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Pink_Assigned_Robot'])
    print("Spearman correlation:", spearman_corr, pval)

    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Green_Assigned_Robot'])
    print("Spearman correlation:", spearman_corr, pval)
    # # Calculate Kendall correlation
    # kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Green_Assigned_Human'])
    # print("Kendall correlation:", kendall_corr, pval)


def lead_follow_time(df):
    ndf = create_data(df)
    spearman_corr, pval = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Collaboration_Time'])
    print("Spearman correlation:", spearman_corr, pval)

    # Calculate Kendall correlation
    kendall_corr, pval = kendalltau(ndf.loc[:, 'Preference'], ndf.loc[:, 'Collaboration_Time'])
    print("Kendall correlation:", kendall_corr, pval)

    plt.hist(ndf.loc[:, 'Collaboration_Time'])
    plt.show()


def plot_correlations(df):
    ndf = create_data(df)
    spearman_corr_hassign, pval_hassign = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Human_Assign'])
    print("Spearman correlation:", spearman_corr_hassign, pval_hassign)

    spearman_corr_rassing, pval_rassign = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Robot_Assign'])
    print("Spearman correlation:", spearman_corr_rassing, pval_rassign)

    spearman_corr_brassign, pval_brassign = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Blue_Assigned_Robot'])
    print("Spearman correlation:", spearman_corr_brassign, pval_brassign)

    spearman_corr_bhassign, pval_bhassign = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Blue_Assigned_Human'])
    print("Spearman correlation:", spearman_corr_bhassign, pval_bhassign)

    spearman_corr_orassign, pval_orassign = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Orange_Assigned_Robot'])
    print("Spearman correlation:", spearman_corr_orassign, pval_orassign)

    spearman_corr_ohassign, pval_ohassign = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Orange_Assigned_Human'])
    print("Spearman correlation:", spearman_corr_ohassign, pval_ohassign)

    spearman_corr_phassign, pval_phassign = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Pink_Assigned_Human'])
    print("Spearman correlation:", spearman_corr_phassign, pval_phassign)

    spearman_corr_ghassign, pval_ghassign = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Green_Assigned_Human'])
    print("Spearman correlation:", spearman_corr_ghassign, pval_ghassign)

    spearman_corr_grassign, pval_grassign = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Green_Assigned_Robot'])
    print("Spearman correlation:", spearman_corr_grassign, pval_grassign)

    spearman_corr_ctime, pval_ctime = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Collaboration_Time'])
    print("Spearman correlation:", spearman_corr_ctime, pval_ctime)

    spearman_corr_rtravel, pval_rtravel = spearmanr(ndf.loc[:, 'Preference'], ndf.loc[:, 'Robot_Travel'])
    print("Spearman correlation:", spearman_corr_rtravel, pval_rtravel)

    spearman_corr_wrong, pval_wrong = spearmanr(ndf.loc[:, 'Robot_Assign'], ndf.loc[:, 'Errors'])
    print("Spearman correlation:", spearman_corr_wrong, pval_wrong)

    # variables = ['# Tasks assigned\n by human', '# Tasks assigned\n to human',
    #              '# Blues assigned\n to human', '# Blues assigned\n to robot',
    #              '# Oranges assigned\n to human', '# Oranges assigned\n to robot',
    #              '# Greens assigned\n to human', '# Greens assigned\n to robot',
    #              '# Pinks assigned\n to robot',
    #              'Robot\'s \n travel distance', 'Collaboration time', '#Tasks assigned\n to human']
    # cor_values = [spearman_corr_hassign, spearman_corr_rassing,
    #               spearman_corr_brassign, spearman_corr_bhassign,
    #               spearman_corr_orassign, spearman_corr_ohassign,
    #               spearman_corr_grassign, spearman_corr_ghassign,
    #               spearman_corr_phassign,
    #               spearman_corr_rtravel, spearman_corr_ctime, spearman_corr_wrong]
    # pvalues = [pval_hassign, pval_rassign,
    #            pval_brassign, pval_bhassign,
    #            pval_orassign, pval_ohassign,
    #            pval_grassign, pval_ghassign,
    #            pval_phassign,
    #            pval_rtravel, pval_ctime, pval_wrong]

    variables = ['# Tasks assigned\n by human', '# Tasks assigned\n to human',
                 '# Blues assigned\n to human', '# Blues assigned\n to robot',
                 '# Oranges assigned\n to human', '# Oranges assigned\n to robot',
                 '# Greens assigned\n to human', '# Greens assigned\n to robot',
                 'Robot\'s \n travel distance']
    cor_values = [spearman_corr_hassign, spearman_corr_rassing,
                  spearman_corr_brassign, spearman_corr_bhassign,
                  spearman_corr_orassign, spearman_corr_ohassign,
                  spearman_corr_grassign, spearman_corr_ghassign,
                  spearman_corr_rtravel]
    pvalues = [pval_hassign, pval_rassign,
               pval_brassign, pval_bhassign,
               pval_orassign, pval_ohassign,
               pval_grassign, pval_ghassign,
               pval_rtravel]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axhline(0, color='red', linestyle='dashed')
    bars = ax.bar(variables, cor_values, color='#607D8B')
    # specific_bar_index = variables.index('#Tasks assigned\n to human')
    # bars[specific_bar_index].set_color('#F39C12')
    for bar, p_value in zip(bars[:-1], pvalues[:-1]):
        bar_height = bar.get_height()
        if bar_height < 0:
            bar_height -= 0.12
        else:
            bar_height += 0.08
        if p_value < 0.001:
            text_p = 'pval<.001'
        else:
            str_p = str(round(p_value, 3))
            text_p = 'pval=' + str_p[1:]

        ax.text(bar.get_x()-0.19, bar_height, text_p, ha='left', va='center',
                fontsize=18, rotation=0)
    ax.text(bars[-1].get_x() - 0.1, 0.1, text_p, ha='left', va='center',
            fontsize=18, rotation=0)

    ax.set_ylim([-1, 1.1])
    ax.set_ylabel(r'Correlation with $oep$', fontsize=25)
    # Correlation with participants\n preference to follow
    # Show the plot
    ax.set_xticklabels(variables, rotation=-80)
    ax.set_yticks(np.arange(-1, 1.2, 0.2))
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.tight_layout()
    # legend_elements = [Rectangle((0, 0), 1, 1, color='#607D8B', ec='black', label='Following preference'),
    #                    Rectangle((0, 0), 1, 1, color='#F39C12', ec='black', label='#Errors')]
    # Add a custom legend
    # plt.legend(handles=legend_elements, loc='upper right')
    # plt.legend(['Regular', 'Special', 'Regular'], loc='upper right')
    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
    plt.savefig('result_follow_taks_correl.eps', format='eps')
    plt.show()


def robot_human_colors(df):
    ndf = create_data(df)
    x1 = ['Blue_Done_Human', 'Orange_Done_Human', 'Pink_Done_Human', 'Green_Done_Human']
    # fig, ax = plt.subplots(1)
    # sns.boxplot(data=ndf[x1], showmeans=True, meanprops=meanprops, **props)
    # ax.set_xticklabels(x1, rotation=-30)
    # plt.tight_layout()
    # plt.show()

    x2 = ['Blue_Done_Robot', 'Orange_Done_Robot', 'Pink_Done_Robot', 'Green_Done_Robot']
    # fig, ax = plt.subplots(1)
    # sns.boxplot(data=ndf[x1], showmeans=True, meanprops=meanprops, **props)
    # ax.set_xticklabels(x1, rotation=-30)
    # plt.tight_layout()
    # plt.show()

    x3 = ['Blue_Assigned_Robot', 'Orange_Assigned_Robot', 'Pink_Assigned_Robot', 'Green_Assigned_Robot']
    # fig, ax = plt.subplots(1)
    # sns.boxplot(data=ndf[x1], showmeans=True, meanprops=meanprops, **props)
    # ax.set_xticklabels(x1, rotation=-30)
    # plt.tight_layout()
    # plt.show()

    x4 = ['Blue_Assigned_Human', 'Orange_Assigned_Human', 'Pink_Assigned_Human', 'Green_Assigned_Human']
    # fig, ax = plt.subplots(1)
    # sns.boxplot(data=ndf[x1], showmeans=True, meanprops=meanprops, **props)
    # ax.set_xticklabels(x1, rotation=-30)
    # plt.tight_layout()
    # plt.show()

    col_data = ndf[x1 + x2 + x3 + x4]
    col_data = pd.melt(col_data)
    col_data['Color'] = ''
    col_data['Action'] = ''
    col_data['Pattern'] = ''
    for i, row in col_data.iterrows():
        if row['variable'] in x1:
            col_data.loc[i, 'Action'] = 'Done_Human'
        elif row['variable'] in x2:
            col_data.loc[i, 'Action'] = 'Done_Robot'
        elif row['variable'] in x3:
            col_data.loc[i, 'Action'] = 'Assigned_Robot'
        elif row['variable'] in x4:
            col_data.loc[i, 'Action'] = 'Assigned_Human'

        if row['variable'] in [x1[0], x2[0], x3[0], x4[0]]:
            col_data.loc[i, 'Color'] = 'Blue'
        elif row['variable'] in [x1[1], x2[1], x3[1], x4[1]]:
            col_data.loc[i, 'Color'] = 'Orange'
        elif row['variable'] in [x1[2], x2[2], x3[2], x4[2]]:
            col_data.loc[i, 'Color'] = 'Pink'
        elif row['variable'] in [x1[3], x2[3], x3[3], x4[3]]:
            col_data.loc[i, 'Color'] = 'Green'
    print(col_data.to_string())
    plt.figure(figsize=(13, 6))
    custom_palette = {'Green': '#009c86', 'Blue': '#1752a2', 'Orange': '#ff8033', 'Pink': '#be587d'}
    g = sns.boxplot(y='value', x='Action', data=col_data, hue='Color',
                    order=[
                        'Done_Human', 'Done_Robot', 'Assigned_Human', 'Assigned_Robot'],
                    fliersize=2, palette=custom_palette, showmeans=True,
                    meanprops=meanprops)

    g.set_xticklabels(['Done by\n Human', 'Done by\n Robot',
                       'Assigned by\n Human', 'Assigned by\n Robot'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim([-0.2, 7])
    g.set_ylabel('#Blocks', fontsize=20)
    g.set_xlabel('Actions', fontsize=20)
    g.set_yticks([0, 1, 2, 3, 4, 5])
    new_labels = ['Blue (Far-Far)', 'Orange (Close-Far)', 'Pink (Far-Close)', 'Green (Close-Close)']
    legend = g.legend(title='Colors (distance: Human-Robot)', loc='upper right', ncol=2, fontsize=16, title_fontsize=16)
    for text, label in zip(legend.texts, new_labels):
        text.set_text(label)

        # Add scatter points to the first four bars
    ax = g.axes
    # Calculate x positions of each box by adjusting for the hue
    xticks = ax.get_xticks()
    width_adjustment = 0.2  # approximate width between bars due to the hue

    # Adjust x positions for the four bars of the first category (Done_Human)
    x_positions = [xticks[0] - width_adjustment * 1.5,  # First bar (Blue)
                   xticks[0] - width_adjustment * 0.5,  # Second bar (Orange)
                   xticks[0] + width_adjustment * 0.5,  # Third bar (Pink)
                   xticks[0] + width_adjustment * 1.5,
                   xticks[1] - width_adjustment * 1.5,  # First bar (Blue)
                   xticks[1] - width_adjustment * 0.5,  # Second bar (Orange)
                   xticks[1] + width_adjustment * 0.5,  # Third bar (Pink)
                   xticks[1] + width_adjustment * 1.5
                   ]  # Fourth bar (Green)
    #
    # # Add scatter points to these positions
    plt.scatter(x_positions[0], 5, marker='s', color='none', s=200, edgecolor='black', zorder=3)
    plt.scatter(x_positions[0]-0.002, 5.01, marker='*', color='black', s=100, edgecolor='black', zorder=3)

    plt.scatter(x_positions[1], 5, marker='s', color='none', s=200, edgecolor='black', zorder=3)
    plt.scatter(x_positions[1]-0.002, 5.01, marker='*', color='black', s=100, edgecolor='black', zorder=3)

    plt.scatter(x_positions[2], 1, marker='s', color='none', s=200, edgecolor='black', zorder=3)
    plt.scatter(x_positions[2]-0.002, 0.01, marker='*', color='black', s=100, edgecolor='black', zorder=3)

    plt.scatter(x_positions[3], 5, marker='s', color='none', s=200, edgecolor='black', zorder=3)
    plt.scatter(x_positions[3]-0.002, 4.01, marker='*', color='black', s=100, edgecolor='black', zorder=3)

    plt.scatter(x_positions[4], 0, marker='s', color='none', s=200, edgecolor='black', zorder=3)
    plt.scatter(x_positions[4]-0.002, 0.01, marker='*', color='black', s=100, edgecolor='black', zorder=3)

    plt.scatter(x_positions[5], 0, marker='s', color='none', s=200, edgecolor='black', zorder=3)
    plt.scatter(x_positions[5]-0.002, 0.01, marker='*', color='black', s=100, edgecolor='black', zorder=3)

    plt.scatter(x_positions[6], 4, marker='s', color='none', s=200, edgecolor='black', zorder=3)
    plt.scatter(x_positions[6]-0.002, 5.01, marker='*', color='black', s=100, edgecolor='black', zorder=3)
    markers = [0, 0]
    markers[0] = plt.scatter(x_positions[7], 0, marker='s', color='none', s=200, edgecolor='black',
                label='Normal-speed human', zorder=3)
    markers[1] = plt.scatter(x_positions[7]-0.002, 1.01, marker='*', color='black', s=100, edgecolor='black',
                label='Low-speed human', zorder=3)

    second_legend = plt.legend(handles=markers, loc='upper left', fontsize=16, frameon = False)
    ax.add_artist(legend)

    plt.tight_layout()
    plt.savefig('result_color_dist.eps', format='eps')
    plt.show()


def lead_follow_by_mode(df):
    ndf = create_data_first(df)
    var_name = 'Preference'
    fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    axs[0].hist(ndf[ndf['Mode'] == 'p1'][var_name])
    axs[1].hist(ndf[ndf['Mode'] == 'p2'][var_name])
    axs[2].hist(ndf[ndf['Mode'] == 'p3'][var_name])
    plt.show()

    fig, ax = plt.subplots(1)
    sns.boxplot(y='Preference', x='Mode', data=ndf, fliersize=3, showmeans=True, meanprops=meanprops, **props)

    result = stats.kruskal(*[group_data['Preference'] for _, group_data in ndf.groupby('Mode')])
    print(result)
    if result.pvalue < 0.05:
        dunn_result = sp.posthoc_dunn(ndf, val_col='Preference', group_col='Mode')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=ndf, dunn_result=dunn_result, fig_ax=ax, ly=1,
                                 pairs=significant_pairs, groups=all_patterns, x='Mode', y='Preference',
                                 y_annotation=0)
    # ax.set_ylim(-0.5, 16)
    # ax.set_yticks(range(0, 13, 2))
    # ax.set_yticklabels(range(0, 13, 2))
    # ax.set_xticklabels(['Pattern 1', 'Pattern 2', 'Pattern 3'])
    ax.set_xlabel('Patterns')
    ax.set_ylabel('Preference')
    plt.tight_layout()
    plt.show()


def robot_human_tasks_done_pattern(df):
    ndf = create_data(df)
    x1 = ['Blue_Done_Human', 'Orange_Done_Human', 'Pink_Done_Human', 'Green_Done_Human']
    x3 = ['Blue_Assigned_Robot', 'Orange_Assigned_Robot', 'Pink_Assigned_Robot', 'Green_Assigned_Robot']

    # fig, ax = plt.subplots(1)
    # sns.boxplot(data=ndf[x1], showmeans=True, meanprops=meanprops, **props)
    # ax.set_xticklabels(x1, rotation=-30)
    # plt.tight_layout()
    # plt.show()

    # x2 = ['Blue_Done_Robot', 'Orange_Done_Robot', 'Pink_Done_Robot', 'Green_Done_Robot']
    x4 = ['Pattern']
    patt_task_data = ndf[x1 + x3 + x4]
    # patt_task_data['overall'] = ''
    patt_task_data.loc[:, 'overall'] = patt_task_data[x1[0]] + patt_task_data[x1[1]] + patt_task_data[x1[2]] + patt_task_data[x1[3]]\
                                       -patt_task_data[x3[0]] - patt_task_data[x3[1]] - patt_task_data[x3[2]] - patt_task_data[x3[3]]
    fig, ax = plt.subplots(1)
    sns.boxplot(data=patt_task_data, y='overall', x='Pattern', fliersize=3, showmeans=True, meanprops=meanprops, **props)
    # ax.set_xticklabels(x1, rotation=-30)
    # plt.tight_layout()


    result = stats.kruskal(*[group_data['overall'] for _, group_data in patt_task_data.groupby('Pattern')])
    print(result)


    if result.pvalue < 0.08:
        dunn_result = sp.posthoc_dunn(patt_task_data, val_col='overall', group_col='Pattern')
        print(dunn_result)

        significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
        au.plot_significant_dunn(data=patt_task_data, dunn_result=dunn_result, fig_ax=ax, ly=1.1,
                                 pairs=significant_pairs, groups=[2,3,4], x='Pattern', y='overall', y_annotation=0)

    ax.set_ylim(-0.5, 18)
    ax.set_xticklabels(['Pattern B', 'Pattern C', 'Pattern D'], fontsize=10)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.set_yticklabels([0, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.set_ylabel('#Subtasks')
    ax.set_xlabel('Patterns', fontsize=13)

    plt.tight_layout()
    plt.savefig('result_assigned_by_h2h.eps', format='eps')
    plt.show()