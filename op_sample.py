import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.polynomial import polyval


tmp_list = [11024]

fig, ax = plt.subplots()
poly = PolynomialFeatures(degree=3, include_bias=False)
poly_reg_model = LinearRegression()
colors = ['red', 'green', 'orange', 'yellow']

for fn in tmp_list:
    for i in range(3,4):
        file_name = 'robot_data/' + str(fn) + '/' + 'task' + str(i) + '.pickle'
        # try:
        rfile = open(file_name, 'rb')
        rdata = pickle.load(rfile, encoding="latin1")

        x_val1 = [x[0] / rdata.p_f[-1][0] for x in rdata.p_f]
        y_val1 = [x[1] for x in rdata.p_f]
        xarr = np.array(x_val1)
        yarr = np.array(y_val1)
        xpoly = poly.fit_transform(xarr.reshape(-1, 1))
        poly_reg_model.fit(xpoly, yarr)
        y_predicted = poly_reg_model.predict(xpoly)
        plt.plot(xarr, y_predicted, color='green', linewidth=2)
        ax.scatter(x_val1, y_val1, linewidth=3, color='red')




ax.set_xlabel('Normalized Time', fontsize=16)
ax.set_ylabel(r'$\alpha_f$', fontsize=20)
# lgd = ax.legend([r'$p_f$', r'$p_e$'], fontsize=15, loc='upper left', frameon=False,
#                 bbox_to_anchor=(0.25, 1), ncol=2)
ax.set_title(r'Bayes estimate of $\alpha_f$', fontsize=16)
ax.set_ylim([0, 1.19])
ax.set_xlim([-0.01, 1.01])
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.tight_layout()
plt.savefig('result_sample_op.eps', format='eps')
plt.show()

