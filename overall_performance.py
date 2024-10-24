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
def plot_overall_perf(df):
    ndf = df.loc[:, ['f1', 'f2', 'f3', 'f4']]
    ndff = pd.melt(ndf)
    ndff['value'] = pd.to_numeric(ndff['value'])
    # fig, ax = plt.subplots(1, tight_layout=True)
    fig = plt.figure()
    sns.boxplot(x="variable", y="value", data=ndff, fliersize=3, showmeans=True, meanprops=meanprops,**props)
    ax = plt.gca()
    ax.set_xticklabels(['Our team improved \n over time',
                        'Our team\'s fluency \n improved over time',
                        'The robot\'s performance \n improved over time',
                        'I will be happy \n to collaborate again', ], rotation=-25, fontsize=8)
    ax.set_ylim(0.5, 7.2)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels([1, 2, 3, 4, 5, 6, 7])
    ax.set_xlabel('')
    ax.set_ylabel('Score')
    plt.tight_layout()
    plt.savefig('result_overal_perfo.eps', format='eps')
    plt.show()