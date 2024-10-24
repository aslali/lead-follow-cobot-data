import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import analyze_utils as au
import seaborn as sns
import data_parameters as prm

ques = ['ad11', 'ad12', 'ad13', 'ad14', 'ad15', 'ad16', 'ad17',
        'ad21', 'ad22', 'ad23', 'ad24', 'ad25', 'ad26', 'ad27',
        'ad31', 'ad32', 'ad33', 'ad34', 'ad35', 'ad36', 'ad37' ]
all_tasks = ['Task 1', 'Task 2', 'Task 3']
all_patterns = [1, 2, 3, 4]


def fetch_human(df):
    n1 = df.loc[:, ques]
    # n1 = nn.rename(columns={ques_time[0]: all_tasks[0], ques_time[1]: all_tasks[1],
    #                         ques_time[2]: all_tasks[2]})

    # fig, axs = plt.subplots(3, sharey=True, tight_layout=True)
    # axs[0].hist(n1[all_tasks[0]])
    # axs[1].hist(n1[all_tasks[1]])
    # axs[2].hist(n1[all_tasks[2]])
    # plt.show()

    n1n = pd.melt(n1.reset_index(), id_vars=['index'], var_name='Ques')
    n1n['Task'] = ''
    n1n['Q'] = ''
    for i, row in n1n.iterrows():
        if row['Ques'] in ['ad11', 'ad12', 'ad13', 'ad14', 'ad15', 'ad16', 'ad17', 'ad18']:
            n1n.loc[i, 'Task'] = 'Task 1'
        elif row['Ques'] in ['ad21', 'ad22', 'ad23', 'ad24', 'ad25', 'ad26', 'ad27', 'ad28']:
            n1n.loc[i, 'Task'] = 'Task 2'
        else:
            n1n.loc[i, 'Task'] = 'Task 3'

        if row['Ques'] in ['ad11', 'ad21', 'ad31']:
            n1n.loc[i, 'Q'] = 'Q1'
        elif row['Ques'] in ['ad12', 'ad22', 'ad32']:
            n1n.loc[i, 'Q'] = 'Q2'
        elif row['Ques'] in ['ad13', 'ad23', 'ad33']:
            n1n.loc[i, 'Q'] = 'Q3'
        elif row['Ques'] in ['ad14', 'ad24', 'ad34']:
            n1n.loc[i, 'Q'] = 'Q4'
        elif row['Ques'] in ['ad15', 'ad25', 'ad35']:
            n1n.loc[i, 'Q'] = 'Q5'
        elif row['Ques'] in ['ad16', 'ad26', 'ad36']:
            n1n.loc[i, 'Q'] = 'Q6'
        elif row['Ques'] in ['ad17', 'ad27', 'ad37']:
            n1n.loc[i, 'Q'] = 'Q7'
        # elif row['Ques'] in ['ad18', 'ad28', 'ad38']:
        #     n1n.loc[i, 'Q'] = 'Q8'

    print(n1n.to_string())
    # nnn = n1n.melt(id_vars=['index'], var_name='Ques')
    # print(nnn.to_string())
    # fig, ax = plt.subplots(1)
    n1n['value'] = 3*(n1n['value'] - 1)/2 + 1
    plt.figure(figsize=(13, 6))
    custom_palette = {'Task 1': '#607D8B', 'Task 2': '#F39C12', 'Task 3': '#9C27B0'}
    g = sns.boxplot(y='value', x='Q', data=n1n, hue='Task', fliersize=2, palette=custom_palette, showmeans=True, meanprops={"marker": "o",
                                                                                       "markerfacecolor": "white",
                                                                                       "markeredgecolor": "black",
                                                                                       "markersize": "8"})


    result = stats.kruskal(*[group_data['value'] for _, group_data in n1n[n1n['Q'] == 'Q1'].groupby('Task')])
    print(result)
    dunn_result = sp.posthoc_dunn(n1n[n1n['Q'] == 'Q1'], val_col='value', group_col='Task')
    print(dunn_result)
    significant_pairs = au.get_significant_pairs_dunn(dunn_result=dunn_result, groups=dunn_result.columns.to_list())
    x1 = -0.25
    x2 = 0.0
    y_annotation = 7
    ly = 0.15
    plt.plot([x1, x1, x2, x2],
             [y_annotation + 0.4 * ly, y_annotation + 1.4 * ly, y_annotation + 1.4 * ly, y_annotation + 0.4 * ly],
             c="black")
    p_val = dunn_result.loc['Task 1', 'Task 2']
    text = 'p={:.3f}'.format(p_val)
    plt.text(x1 - 0.45, y_annotation + 0.5 * ly, text, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))
    y_annotation += 1.5 * ly
    x2 = 0.25
    p_val = dunn_result.loc['Task 1', 'Task 3']
    text = 'p={:.3f}'.format(p_val)
    plt.plot([x1, x1, x2, x2],
             [y_annotation + 0.4 * ly, y_annotation + 1.4 * ly, y_annotation + 1.4 * ly, y_annotation + 0.4 * ly],
             c="black")
    plt.text(x2 + 0.05, y_annotation + 1.0 * ly, text, fontsize=10,
             bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))

    g.set_xticklabels(['Inteligence', 'Commitment\n to task', 'Perceiving\n my goals',
                       'Not understanding \n what I want to do',
                       'Working towards\n mutual goals', 'Respecting\n each other',
                       'Appreciating me'], rotation=-25, fontsize=10)

    plt.ylabel('Score')
    plt.tight_layout()
    plt.xlabel('')
    plt.savefig('result_traits.eps', format='eps')
    plt.show()
