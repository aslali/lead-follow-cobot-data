import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.polynomial import polyval



# tmp_list = [15580]
#
# fig = plt.figure()
# ax = plt.gca()
# poly = PolynomialFeatures(degree=3, include_bias=False)
# poly_reg_model = LinearRegression()
# colors = ['red', 'green', 'orange', 'yellow']
#
# for fn in tmp_list:
#     for i in range(4,5):
#         file_name = 'robot_data/' + str(fn) + '/' + 'task' + str(i) + '.pickle'
#         # try:
#         rfile = open(file_name, 'rb')
#         rdata = pickle.load(rfile, encoding="latin1")
#
#         x_val1 = [x[0] / rdata.p_f[-1][0] for x in rdata.p_f]
#         y_val1 = [x[1] for x in rdata.p_f]
#         x_val2 = [x[0] / rdata.p_f[-1][0] for x in rdata.p_e]
#         y_val2 = [x[1] for x in rdata.p_e]
#         ax.scatter(x_val1, y_val1, linewidth=3, color='red')
#         ax.scatter(x_val2, y_val2, linewidth=3, color='green')
#         xarr = np.array(x_val1)
#         yarr = np.array(y_val1)
#         plt.plot(x_val1, y_val1, color='red')
#         plt.plot(x_val2, y_val2, color='green')
#
#
#
#
#
# ax.set_xlabel('Normalized Time', fontsize=16)
# ax.set_ylabel(r'$\alpha_f, \alpha_e$', fontsize=20)
# lgd = ax.legend([r'$\alpha_f$: Following Preference', r'$\alpha_e$: Error-proneness'],
#                 fontsize=12, loc='upper left', frameon=False,
#                 ncol=1)
# bbox_to_anchor=(0.25, 1)
# ax.set_title(r'Bayes estimate of $\alpha_f$ and $\alpha_e$', fontsize=16)
# ax.set_ylim([0, 1.19])
# ax.set_xlim([-0.01, 1.01])
# ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# ax.tick_params(axis='x', labelsize=16)
# ax.tick_params(axis='y', labelsize=16)
# plt.tight_layout()
# plt.savefig('result_cases_ope4_15580.eps', format='eps')
# plt.show()






# id_list =[10070,
# 10181,
# 10692,
# 10810,
# 11024,
# 11520,
# 11671,
# 11810,
# 12589,
# 12772,
# 13179,
# 14108,
# 14336,
# 14391,
# 14416,
# 14723,
# 14974,
# 15048,
# 15164,
# 15232,
# 15580,
# 15638,
# 15907,
# 16268,
# 16447,
# 16631,
# 16990,
# 17100,
# 17478,
# 17619,
# 17652,
# 17748,
# 17820,
# 17875,
# 18000,
# 18261,
# 18285,
# 18327,
# 18647,
# 18657,
# 18794,
# 19032,
# 19138,
# 19298,
# 19538,
# 19551,
# 19597,
# 19888]
task_order = {14391: [3, 2, 4], 18261: [2, 4, 3], 17820: [4, 2, 3],
              15048: [3, 4, 2], 14974: [2, 3, 4], 19032: [3, 2, 4],
              11671: [2, 4, 3], 17748: [2, 3, 4], 14336: [4, 3, 2],
              13179: [3, 4, 2]}
id_list = [13179]
for ii in id_list:
    tmp_list = [ii]
    fig = plt.figure()
    ax = plt.gca()
    poly = PolynomialFeatures(degree=3, include_bias=False)
    poly_reg_model = LinearRegression()
    colors = ['red', 'green', 'orange', 'yellow']
    # linestyles= {3: 'solid', 2: 'dashed', 4: 'dotted'}
    # linestyles = {2: 'solid', 4: 'dashed', 3: 'dotted'}
    linestyles = {2: 'solid', 3: 'dashed', 4: 'dotted'}
    figs_p = []
    figs_e = []
    for fn in tmp_list:
        for k in [3]: #[2, 3, 4]:
            i = task_order[fn].index(k) + 2
            file_name = 'robot_data/' + str(fn) + '/' + 'task' + str(i) + '.pickle'
            # try:
            rfile = open(file_name, 'rb')
            rdata = pickle.load(rfile, encoding="latin1")

            x_val1 = [x[0] / rdata.p_f[-1][0] for x in rdata.p_f]
            y_val1 = [x[1] for x in rdata.p_f]
            x_val2 = [x[0] / rdata.p_f[-1][0] for x in rdata.p_e]
            y_val2 = [x[1] for x in rdata.p_e]
            ax.scatter(x_val1, y_val1, linewidth=3, color='red')
            ax.scatter(x_val2, y_val2, linewidth=3, color='green')
            xarr = np.array(x_val1)
            yarr = np.array(y_val1)
            fig1, = plt.plot(x_val1, y_val1, linestyle=linestyles[k], color='red')
            fig2, = plt.plot(x_val2, y_val2, linestyle=linestyles[k], color='green')
            figs_p.append(fig1)
            figs_e.append(fig2)




    ax.set_xlabel('Normalized Time', fontsize=20)
    ax.set_ylabel(r'$E(\alpha_f), E(\alpha_e)$', fontsize=20)
    # legend1 = plt.legend(figs_p, ["Pattern B", "Pattern C", "Pattern D"], loc=1, fontsize=14)
    legend1 = plt.legend(figs_p, ["Pattern C"], loc=1, fontsize=14)
    lgd = ax.legend([r'$\alpha_f$: Following Preference', r'$\alpha_e$: Error-proneness'],
                    fontsize=15, loc='upper left', frameon=False,
                    ncol=1)
    for legline in legend1.get_lines():
        legline.set_color('black')
    bbox_to_anchor = (0.25, 1)
    ax.set_title(r'Expected $\alpha_f$ and $\alpha_e$ over time', fontsize=20)
    ax.set_ylim([0, 1.19])
    ax.set_xlim([-0.01, 1.01])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    ax.add_artist(legend1)
    # plt.text(0.5, 0.5, str(ii))
    plt.tight_layout()

    plt.savefig('result_cases_ope4_13179.eps', format='eps')
    plt.show()