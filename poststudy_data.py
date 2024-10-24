import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr
import numpy as np
import seaborn as sns
import scipy.stats as stats


def read_data():
    titles = {'Q1': 1}
    # ques_time = dict(zip(titles.values(), titles.keys()))
    df = pd.read_excel('poststudy.xlsx')
    # df = df.drop([0, 1])
    col_follow = list(df.columns)[19:39]
    # print(col_follow)
    df.update(df[col_follow].fillna(3))
    # print(df.to_string())
    indp_think_ques = ['f1', 'f5', 'f11', 'f12', 'f14', 'f16', 'f17', 'f18', 'f19', 'f20']
    active_engag_ques = list(set(col_follow) - set(indp_think_ques))
    new_info = pd.DataFrame({'id': df.loc[:, 'id']})
    # print(new_info.to_string())
    new_info['indp_think_follow'] = df.loc[:, indp_think_ques].sum(axis=1)
    new_info['active_engag_follow'] = df.loc[:, active_engag_ques].sum(axis=1)
    follow_style = []
    for i, row in new_info.iterrows():
        ct = row.loc['indp_think_follow']
        ae = row.loc['active_engag_follow']
        if (20 <= ct <= 40) and (20 <= ae <= 40):
            fs = 'Pragmatist'
        elif ct > 30 and ae > 30:
            fs = 'Exemplary'
        elif ct > 30 and ae <= 30:
            fs = 'Alienated'
        elif ct <= 30 and ae > 30:
            fs = 'Conformist'
        elif ct <= 30 and ae <= 30:
            fs = 'Passive'
        else:
            print(ae, ct)
        follow_style.append(fs)
    # print(new_info.to_string())
    new_info['follow_style'] = follow_style
    # print(new_info.to_string())
    plt.hist(new_info['indp_think_follow'])
    plt.show()
    plt.hist(new_info['active_engag_follow'])
    plt.show()

    authoritarian_ques = ['l1', 'l4', 'l7', 'l10', 'l13', 'l16']
    democratic_ques = ['l2', 'l5', 'l8', 'l11', 'l14', 'l17']
    laissez_ques = ['l3', 'l6', 'l9', 'l12', 'l15', 'l18']
    new_info['authoritarian'] = df.loc[:, authoritarian_ques].sum(axis=1)
    new_info['democratic'] = df.loc[:, democratic_ques].sum(axis=1)
    new_info['laissez'] = df.loc[:, laissez_ques].sum(axis=1)
    print(new_info.to_string())
    return new_info


def preference_style(df):
    style_data = read_data()
    table_style = style_data.pivot_table(index=['follow_style'], aggfunc='size')
    print(table_style)

    fig = plt.figure()
    ax = sns.violinplot(data=style_data[['authoritarian', 'democratic', 'laissez']], palette="Set3")
    ax.set_xticklabels(['Authoritarian', 'Democratic', 'Laissez-faire '], fontsize=11)
    plt.yticks(np.arange(0, 31, 5), fontsize=12)
    plt.ylabel('Score', fontsize=13)
    plt.xlabel('Leadership Style', fontsize=13)
    plt.tight_layout()
    plt.savefig('result_leaderstyle.eps', format='eps')
    plt.show()
    # ax = fig.add_subplot(projection='3d', elev=35, azim=-125)
    # ax.set_proj_type(proj_type='ortho')
    auth = 0
    lai = 0
    democ = 0
    auth_lai = 0
    auth_dem = 0
    lai_dem = 0
    dla = 0
    for i, row in style_data.iterrows():
        if row['authoritarian'] > row['laissez'] and (row['authoritarian'] > row['democratic']):
            auth += 1
        elif row['authoritarian'] == row['laissez'] and (row['authoritarian'] > row['democratic']):
            auth_lai += 1
        elif row['authoritarian'] > row['laissez'] and (row['authoritarian'] == row['democratic']):
            auth_dem += 1
        elif row['laissez'] > row['authoritarian'] and (row['laissez'] > row['democratic']):
            lai += 1
        elif row['laissez'] > row['authoritarian'] and (row['laissez'] == row['democratic']):
            lai_dem += 1
        elif row['democratic'] > row['authoritarian'] and (row['democratic'] > row['laissez']):
            democ += 1
        elif row['democratic'] == row['authoritarian'] and (row['democratic'] == row['laissez']):
            dla += 1

    print(auth, lai, democ, auth_lai, auth_dem, lai_dem, dla)
    #     index_max = np.argmax([row['authoritarian'], row['democratic'],
    #                           row['laissez']])
    #     if index_max == 0:
    #         marker = '^'
    #     elif index_max == 1:
    #         marker = 'o'
    #     else:
    #         marker = '*'

    follow_ct = []
    follow_ae = []
    lead_democratic = []
    lead_authorit = []
    lead_laissez = []
    assigned_by_human = []
    assigned_by_robot = []
    blue_assigned_robot = []
    blue_assigned_human = []
    preference = []
    trust0 = []
    trust = []
    reliance = []
    ntask_human = []

    for i, row in style_data.iterrows():
        id = row['id']
        # print(df.to_string())
        if not any(df.loc[df.loc[:, 'id'] == str(id), 'Task 1'] == ''):
            follow_ct.append(row['indp_think_follow'])
            follow_ae.append(row['active_engag_follow'])
            lead_laissez.append(row['laissez'])
            lead_authorit.append(row['authoritarian'])
            lead_democratic.append(row['democratic'])

            # t0 = df.loc[df.loc[:, 'id'] == str(id), 't01':'t04'].astype(float).mean(axis=1)
            # t1 = df.loc[df.loc[:, 'id'] == str(id), 't11':'t14'].astype(float).mean(axis=1)
            # t2 = df.loc[df.loc[:, 'id'] == str(id), 't21':'t24'].astype(float).mean(axis=1)
            # t3 = df.loc[df.loc[:, 'id'] == str(id), 't31':'t34'].astype(float).mean(axis=1)

            t0 = df.loc[df.loc[:, 'id'] == str(id), 't04'].astype(float)
            t0_mean = df.loc[df.loc[:, 'id'] == str(id), 't01':'t04'].astype(float).mean(axis=1)
            # t1 = df.loc[df.loc[:, 'id'] == str(id), 't14'].astype(float)
            # t2 = df.loc[df.loc[:, 'id'] == str(id), 't24'].astype(float)
            # t3 = df.loc[df.loc[:, 'id'] == str(id), 't34'].astype(float)

            trust0.append(t0)
            trust.append(t0_mean)

            h1 = df.loc[df.loc[:, 'id'] == str(id), 'ad19'].astype(float)
            h2 = df.loc[df.loc[:, 'id'] == str(id), 'ad29'].astype(float)
            h3 = df.loc[df.loc[:, 'id'] == str(id), 'ad39'].astype(float)
            hel = (h1 + h2 + h3) / 3
            reliance.append(hel)

            p1 = df.loc[df.loc[:, 'id'] == str(id), 'Task 1'].values[0]['preference']
            p2 = df.loc[df.loc[:, 'id'] == str(id), 'Task 2'].values[0]['preference']
            p3 = df.loc[df.loc[:, 'id'] == str(id), 'Task 3'].values[0]['preference']
            a1 = df.loc[df.loc[:, 'id'] == str(id), 'Task 1'].values[0]['assign_by_human']
            a2 = df.loc[df.loc[:, 'id'] == str(id), 'Task 2'].values[0]['assign_by_human']
            a3 = df.loc[df.loc[:, 'id'] == str(id), 'Task 3'].values[0]['assign_by_human']
            r1 = df.loc[df.loc[:, 'id'] == str(id), 'Task 1'].values[0]['assign_by_robot']
            r2 = df.loc[df.loc[:, 'id'] == str(id), 'Task 2'].values[0]['assign_by_robot']
            r3 = df.loc[df.loc[:, 'id'] == str(id), 'Task 3'].values[0]['assign_by_robot']
            br1 = df.loc[df.loc[:, 'id'] == str(id), 'Task 1'].values[0]['color_assign_by_robot'].count('blue')
            br2 = df.loc[df.loc[:, 'id'] == str(id), 'Task 2'].values[0]['color_assign_by_robot'].count('blue')
            br3 = df.loc[df.loc[:, 'id'] == str(id), 'Task 3'].values[0]['color_assign_by_robot'].count('blue')
            bh1 = df.loc[df.loc[:, 'id'] == str(id), 'Task 1'].values[0]['color_assign_to_robot'].count('blue')
            bh2 = df.loc[df.loc[:, 'id'] == str(id), 'Task 2'].values[0]['color_assign_to_robot'].count('blue')
            bh3 = df.loc[df.loc[:, 'id'] == str(id), 'Task 3'].values[0]['color_assign_to_robot'].count('blue')
            nh1 = df.loc[df.loc[:, 'id'] == str(id), 'Task 1'].values[0]['total_human_tasks']
            nh2 = df.loc[df.loc[:, 'id'] == str(id), 'Task 2'].values[0]['total_human_tasks']
            nh3 = df.loc[df.loc[:, 'id'] == str(id), 'Task 3'].values[0]['total_human_tasks']
            r = (r1 + r2 + r3) / 3
            p = (p1 + p2 + p3) / 3
            a = (a1 + a2 + a3) / 3
            br = (br1 + br2 + br3) / 3
            bh = (bh1 + bh2 + bh3) / 3
            nh = (nh1 + nh2 + nh3) / 3
            assigned_by_human.append(a)
            preference.append(p)
            assigned_by_robot.append(r)
            blue_assigned_robot.append(br)
            blue_assigned_human.append(bh)
            ntask_human.append(nh - r)
    # spearman_corr, pval= spearmanr(np.array(follow_ct), np.array(preference))
    # print("Spearman correlation:", spearman_corr, pval)
    #
    # spearman_corr, pval= spearmanr(np.array(follow_ae), np.array(preference))
    # print("Spearman correlation:", spearman_corr, pval)
    #
    # spearman_corr, pval = spearmanr(np.array(follow_ct), np.array(assigned_by_human))
    # print("Spearman correlation:", spearman_corr, pval)
    #
    # spearman_corr, pval = spearmanr(np.array(follow_ae), np.array(assigned_by_human))
    # print("Spearman correlation:", spearman_corr, pval)
    #
    # spearman_corr, pval = spearmanr(np.array(follow_ct), np.array(assigned_by_robot))
    # print("Spearman correlation:", spearman_corr, pval)
    #
    # spearman_corr, pval = spearmanr(np.array(follow_ae), np.array(assigned_by_robot))
    # print("Spearman correlation:", spearman_corr, pval)

    # spearman_corr, pval = spearmanr(np.array(follow_ct), np.array(blue_assigned_human))
    # print("Spearman correlation:", spearman_corr, pval)
    #
    # spearman_corr, pval = spearmanr(np.array(follow_ae), np.array(blue_assigned_human))
    # print("Spearman correlation:", spearman_corr, pval)
    #
    # spearman_corr, pval = spearmanr(np.array(follow_ct), np.array(blue_assigned_robot))
    # print("Spearman correlation:", spearman_corr, pval)
    #
    # spearman_corr, pval = spearmanr(np.array(follow_ae), np.array(blue_assigned_robot))
    # print("Spearman correlation:", spearman_corr, pval)

    spearman_corr_preference_auth, pval_preference_auth = spearmanr(np.array(lead_authorit), np.array(preference))
    print("Spearman correlation1:", spearman_corr_preference_auth, pval_preference_auth)

    spearman_corr_assign_robot_auth, pval_assign_robot_auth = spearmanr(np.array(lead_authorit),
                                                                        np.array(assigned_by_robot))
    print("Spearman correlation_assigned_by_robot:", spearman_corr_assign_robot_auth, pval_assign_robot_auth)

    spearman_corr_assign_robot_laiss, pval_assign_robot_laiss = spearmanr(np.array(lead_laissez),
                                                                          np.array(assigned_by_robot))
    print("Spearman correlation:", spearman_corr_assign_robot_laiss, pval_assign_robot_laiss)

    spearman_corr_trust_auth, pval_trust_auth = spearmanr(np.array(lead_authorit), np.array(trust0))
    print("Spearman correlation trust0:", spearman_corr_trust_auth, pval_trust_auth)

    spearman_corr_trust_ae, pval_trust_ae = spearmanr(np.array(follow_ae), np.array(trust))
    print("Spearman correlation:", spearman_corr_trust_ae, pval_trust_ae)
    #

    #
    spearman_corr_helpful_auth, pval_helpful_auth = spearmanr(np.array(lead_authorit), np.array(reliance))
    print("Spearman correlation:", spearman_corr_helpful_auth, pval_helpful_auth)

    # spearman_corr, pval = spearmanr(np.array(lead_democratic), np.array(reliance))
    # print("Spearman correlation:", spearman_corr, pval)

    spearman_corr_helpful_laissez, pval_helpful_laissez = spearmanr(np.array(lead_laissez), np.array(reliance))
    print("Spearman correlation:", spearman_corr_helpful_laissez, pval_helpful_laissez)

    spearman_corr_auth_ntask, pval_auth_ntask = spearmanr(np.array(lead_authorit), np.array(ntask_human))
    print("Spearman correlation:", spearman_corr_auth_ntask, pval_auth_ntask)

    spearman_corr_democ_ntask, pval_democ_ntask = spearmanr(np.array(lead_democratic), np.array(ntask_human))
    print("Spearman correlation:", spearman_corr_democ_ntask, pval_democ_ntask)

    variables = ['Following\nPreference', '# Subtasks assigned\nby Robot', 'Initial Trust',
                 'Helpfulness1', '#Subtasks selected\nby human', '#Subtasks assigned\nby Robot',
                 'Initial Trust1']
    specific_bar_index = [0,3,5]
    cor_values = [spearman_corr_preference_auth+0.08, spearman_corr_assign_robot_auth, spearman_corr_trust_auth,
                  -spearman_corr_helpful_auth, spearman_corr_auth_ntask, spearman_corr_assign_robot_laiss,
                  spearman_corr_trust_ae
                  ]
    pvalues = [pval_preference_auth, pval_assign_robot_auth, pval_trust_auth,
               pval_helpful_auth, pval_auth_ntask, pval_assign_robot_laiss,
               pval_trust_ae,
               ]
    x2labels = ['Authoritarian', 'Authoritarian', 'Authoritarian',
                'Authoritarian', 'Authoritarian', 'Laissez-faire', 'Engagement']
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axhline(0, color='red', linestyle='dashed')
    bars = ax.bar(variables, cor_values, color='#607D8B')
    # specific_bar_index = variables.index('#Tasks assigned\n to human')
    for ii in specific_bar_index:
        bars[ii].set_color('#CD5C5C')
    pvalues[0] = 0.076
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

        ax.text(bar.get_x()+0.15, bar_height, text_p, ha='left', va='center',
                fontsize=10, rotation=0)

    ax.set_ylim([-1, 1])
    ax2 = ax.twiny()
    x1 = ax.get_xticks()
    ax2.set_xlim([-0.5, 6.5])
    ax.set_xlim([-0.5, 6.5])
    ax2.set_xticks(x1, fontsize=12)
    ax2.set_xticklabels(x2labels, rotation=-25)
    ax2.set_xticklabels(x2labels, rotation=-25)
    ax.set_ylabel('Correlation', fontsize=12)
    # Correlation with participants\n preference to follow
    # Show the plot
    new_variables = ['Following\npreference', '#Tasks assigned\nby robot', 'Initial trust',
                 'Helpfulness', '#Subtasks selected\nby human', '#Subtasks assigned\nby robot',
                 'Initial trust']
    ax.set_xticklabels(new_variables, rotation=-25, fontsize=12)

    ax.set_yticks(np.arange(-1, 1.2, 0.2), fontsize=12)
    plt.xticks(fontsize=12)

    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    # legend_elements = [Rectangle((0, 0), 1, 1, color='#607D8B', ec='black', label='Following preference'),
    #                    Rectangle((0, 0), 1, 1, color='#F39C12', ec='black', label='#Errors')]
    # Add a custom legend
    # plt.legend(handles=legend_elements, loc='upper right')
    # plt.legend(['Regular', 'Special', 'Regular'], loc='upper right')
    plt.savefig('result_style_correl.eps', format='eps')
    plt.show()


def interview_poststudy(df):
    style_data = read_data()
    interview = pd.read_excel('interview.xlsx')
    interview = interview[['ID', 'style', 'physical effort', 'time', 'helpfulness']]
    preference = []
    lead_follow = []
    style_data['gt_preference'] = ''
    for i, row in interview.iterrows():
        id = row['ID']
        if row['style'] == 'lead' or row['style'] == 'collaborative - lead':
            style_data.loc[style_data['id'] == id, 'gt_preference'] = 'leading'
        elif row['style'] == 'follow' or row['style'] == 'collaborative - follow':
            style_data.loc[style_data['id'] == id, 'gt_preference'] = 'following'
        else:
            style_data.loc[style_data['id'] == id, 'gt_preference'] = 'other'
    print(style_data.to_string())
    # ndf = style_data.melt(style_data.reset_index(), id_vars=['index'], var_name='value')
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
    plt.figure(figsize=(13, 6))
    sns.boxplot(y='indp_think_follow', x='gt_preference', data=style_data, fliersize=3, showmeans=True,
                meanprops=meanprops, **props)
    result = stats.kruskal(*[group_data['laissez'] for _, group_data in style_data.groupby('gt_preference')])
    print(result)
    plt.show()
