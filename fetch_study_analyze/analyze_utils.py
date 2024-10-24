import matplotlib.pyplot as plt

def get_significant_pairs_dunn(dunn_result, groups, alpha=0.05):
    significant_pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = groups[i]
            group2 = groups[j]
            p_value = dunn_result.loc[group1, group2]
            if p_value < alpha:
                significant_pairs.append((group1, group2))

    return significant_pairs


def plot_significant_dunn(data, dunn_result, fig_ax, pairs, groups, x, y, y_annotation=0, xth=0, ly=1.5):
    xval = fig_ax.get_xticks()
    for pairs in pairs:
        group1, group2 = pairs[0], pairs[1]
        x1 = xval[groups.index(group1)]
        x2 = xval[groups.index(group2)]
        y_annotation1 = data[data[x] == group1].max(axis=0)
        y_annotation2 = data[data[x] == group2].max(axis=0)
        y_annotation = max(y_annotation1[y], y_annotation2[y], y_annotation + 1.5 * ly)
        p_val = dunn_result.loc[group1, group2]
        if p_val < 0.001:
            text = 'p<.001'
        else:
            str_p = str(round(p_val, 3))
            text = 'p=' + str_p[1:]
        # text = 'p={:.3f}'.format(p_val)
        props = {'connectionstyle': 'arc', 'arrowstyle': '-', 'shrinkA': 20, 'shrinkB': 20, 'linewidth': 1}
        plt.plot([x1 + xth, x1 + xth, x2 + xth, x2 + xth], [y_annotation + 0.4 * ly, y_annotation + 1.4 * ly, y_annotation + 1.4 * ly, y_annotation + 0.4 * ly],
                 c="black")
        # fig_ax.annotate(text, xy=(0.45 * (x1 + x2 + 2*xth)- 0.02, y_annotation + 0.3), zorder=2)
        fig_ax.text(0.45 * (x1 + x2 + 2 * xth) - 0.02, y_annotation + 1.2 * ly, text, fontsize=23, ha='center',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))
    return y_annotation