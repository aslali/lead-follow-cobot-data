import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

props = {
    'boxprops': {'facecolor': 'none', 'edgecolor': 'red'},
    'medianprops': {'color': 'green'},
    'whiskerprops': {'color': 'blue'},
    'capprops': {'color': 'blue'},
}

meanprops = {"marker": "o",
             "markerfacecolor": "white",
             "markeredgecolor": "black",
             "markersize": "8"}


def interview_lead_follow(df):
    interview = pd.read_excel('interview.xlsx')
    interview = interview[['ID', 'style', 'physical effort', 'time', 'helpfulness']]
    preference = []
    lead_follow = []

    for i, row in interview.iterrows():
        id = row['ID']
        # print(df.to_string())
        if not any(df.loc[df.loc[:, 'id'] == str(id), 'Task 1'] == ''):
            p1 = df.loc[df.loc[:, 'id'] == str(id), 'Task 1'].values[0]['preference']
            p2 = df.loc[df.loc[:, 'id'] == str(id), 'Task 2'].values[0]['preference']
            p3 = df.loc[df.loc[:, 'id'] == str(id), 'Task 3'].values[0]['preference']
            p = (p1 + p2 + p3) / 3
            print(p)
            preference.append(p)
            lead_follow.append(row['style'])
    pd_lead_follow = pd.DataFrame({'Preference': preference, 'Lead_Follow': lead_follow})

    plt.figure(figsize=(13, 6))
    g = sns.boxplot(y='Preference', x='Lead_Follow', order=['lead', 'collaborative - lead',
                                                            'collaborative - follow', 'follow', 'neither-collaborative',
                                                            'neither-follow'],
                    data=pd_lead_follow, fliersize=3, gap=0.1, widths=0.3, showmeans=True, linewidth=2, meanprops=meanprops,
                    **props)
    df1 = pd_lead_follow.groupby('Lead_Follow')['Preference'].agg(Max='max')
    df2 = pd_lead_follow.groupby('Lead_Follow')['Lead_Follow'].agg(size='size')

    df1 = df1.reset_index()
    df2 = df2.reset_index()
    print(df2)
    for xtick in g.get_xticklabels():
        g.text(xtick.get_position()[0], df1.loc[df1['Lead_Follow'] == xtick.get_text(), 'Max'] + 0.03,
               'count=' + str(df2.loc[df2['Lead_Follow'] == xtick.get_text(), 'size'].iloc[0]),
               horizontalalignment='center', size=20, color='b', weight='semibold')
    ax = plt.gca()
    ax.add_patch(Rectangle((3.5, 0.03), 2, 0.67, edgecolor='black', linewidth=3, linestyle='--',
                           facecolor='none', zorder=-1))
    plt.text(3.9, 0.73, "Outlier participants", size=15, color='black', weight='semibold')
    ax.set_xticklabels(['l', 'Ca', 'b', 'd', 'f', 'N'],
                       fontsize=14, rotation=30)
    plt.xticks( fontsize=16, rotation=10)
    plt.yticks(fontsize=16)
    plt.ylim((0, 0.79))
    plt.xlim((-0.5, 5.6))
    plt.xlabel('Actual Leading/Following Preference', fontsize=20)
    plt.ylabel(r'Overall Estimated Preference ($oep$)', fontsize=20)
    plt.tight_layout()
    # plt.savefig('result_real_estimate.eps', format='eps')
    plt.show()

def interview_lead_follow2(df):
    medianprops = dict(linestyle='-', linewidth=3, color='green')
    meanprops = {'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 8}
    props = {
        'whiskerprops': {'linewidth': 3},  # Adjust whisker line width
        'capprops': {'linewidth': 3},  # Adjust cap line width
    }
    # Load interview data and extract relevant columns
    interview = pd.read_excel('interview.xlsx')
    interview = interview[['ID', 'style', 'physical effort', 'time', 'helpfulness']]
    preference = []
    lead_follow = []

    # Iterate over interview data and gather preferences
    for i, row in interview.iterrows():
        id = row['ID']
        if not any(df.loc[df.loc[:, 'id'] == str(id), 'Task 1'] == ''):
            p1 = df.loc[df.loc[:, 'id'] == str(id), 'Task 1'].values[0]['preference']
            p2 = df.loc[df.loc[:, 'id'] == str(id), 'Task 2'].values[0]['preference']
            p3 = df.loc[df.loc[:, 'id'] == str(id), 'Task 3'].values[0]['preference']
            p = (p1 + p2 + p3) / 3
            preference.append(p)
            lead_follow.append(row['style'])

    # Create DataFrame for plotting
    pd_lead_follow = pd.DataFrame({'Preference': preference, 'Lead_Follow': lead_follow})

    # Order categories manually
    categories = ['lead', 'collaborative - lead', 'collaborative - follow', 'follow', 'neither-collaborative',
                  'neither-follow']
    pd_lead_follow['Lead_Follow'] = pd.Categorical(pd_lead_follow['Lead_Follow'], categories=categories, ordered=True)

    # Prepare data for boxplot
    data = [pd_lead_follow[pd_lead_follow['Lead_Follow'] == category]['Preference'] for category in categories]

    # Create boxplot using matplotlib
    plt.figure(figsize=(12, 8))
    widths = 0.08
    positions = list(np.arange(0.2, 0.24+ 5*(widths+0.05), widths+0.05))
    print(positions)
    boxplot = plt.boxplot(data,
                          positions=positions,
                          patch_artist=True,
                          showmeans=True,
                          widths=widths,
                          meanline=False,
                          showcaps=True,
                          medianprops=medianprops,
                          meanprops=meanprops,
                          **props)

    # Customize colors for each box
    for patch in boxplot['boxes']:
        patch.set_facecolor('none')
        patch.set_edgecolor('red')
        patch.set_linewidth(3)  # Set the line width of the boxes to 2

    # Customize cap size
    for cap in boxplot['caps']:
        cap.set_linewidth(3)  # Adjust linewidth as needed

    # Add custom rectangle annotation
    ax = plt.gca()
    ax.add_patch(
        Rectangle((0.66, 0.03), .25, 0.67, edgecolor='black', linewidth=3, linestyle='--', facecolor='none', zorder=-1))
    plt.text(0.65, 0.73, "Outlier participants", size=22, color='black', weight='semibold')

    # Customize ticks and labels
    plt.xticks(positions, labels=['L', 'CL', 'CF', 'F', 'NC','NF'],
               fontsize=22, weight='semibold')
    plt.yticks(fontsize=22, weight='semibold')
    plt.ylim((0, 0.79))
    plt.xlim((0.12,0.92))
    plt.xlabel('Actual Leading/Following Preference', fontsize=25)
    plt.ylabel(r'Overall Estimated Preference ($oep$)', fontsize=25)

    # Add participant counts
    df1 = pd_lead_follow.groupby('Lead_Follow')['Preference'].agg(Max='max').reset_index()
    df2 = pd_lead_follow.groupby('Lead_Follow')['Lead_Follow'].agg(size='size').reset_index()

    for i, category in enumerate(categories):
        max_val = df1[df1['Lead_Follow'] == category]['Max'].values[0]
        size = df2[df2['Lead_Follow'] == category]['size'].values[0]
        plt.text(positions[i], max_val + 0.03, f'count={size}', horizontalalignment='center', size=22, color='b',
                 weight='semibold')

    plt.tight_layout()
    plt.savefig('result_real_estimate.eps', format='eps')
    plt.show()