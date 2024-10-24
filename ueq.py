import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

def plot_ueq(df):
    ndf = df.loc[:, ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8']]
    ndff = pd.melt(ndf)
    ndff['value'] = pd.to_numeric(ndff['value'])
    # fig, ax = plt.subplots(1, tight_layout=True)
    fig = plt.figure()
    ax = plt.gca()
    sns.boxplot(x="variable", y="value", data=ndff, fliersize=3, showmeans=True, meanprops=meanprops, **props)
    ax.set_xticklabels(['obstructive', 'complicated', 'inefficient', 'confusing', 'boring', 'not \n interesting',
                        'conventional', 'usual'], rotation=-25)
    ax2 = ax.twiny()
    x1 = ax.get_xticks()
    ax2.set_xlim([-0.5, 7.5])
    ax2.set_xticks(x1)
    ax2.set_xticklabels(['supportive', 'easy', 'efficient', 'clear', 'exciting',
                         'interesting', 'inventive', 'leading\n edge'], rotation=-25)
    ax2.set_ylim(0.8, 7.2)
    ax.set_xlabel('')
    ax.set_ylabel('Score')
    plt.tight_layout()
    fig.savefig('result_ueq.eps', format='eps')
    plt.show()
