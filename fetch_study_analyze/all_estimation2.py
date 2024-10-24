import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.polynomial import polyval


tmp_list = [11810, 16631, 16990, 16447, 15232, 18261, 14723, 17619, 17100, 19138, 17748, 14416, 13179, 18794, 19298,
            17875, 19597, 18657, 19538, 15164, 12772, 18327, 15907, 10070, 15048, 10810, 14336]

tmp_list = [10692]

fig, ax = plt.subplots()
poly = PolynomialFeatures(degree=3, include_bias=False)
poly_reg_model = LinearRegression()
colors = ['red', 'green', 'orange', 'yellow']

for fn in tmp_list:
    for i in range(2,5):
        file_name = 'robot_data/' + str(fn) + '/' + 'task' + str(i) + '.pickle'
        # try:
        rfile = open(file_name, 'rb')
        rdata = pickle.load(rfile, encoding="latin1")

        x_val1 = [x[0] / rdata.p_f[-1][0] for x in rdata.p_f]
        y_val1 = [x[1] for x in rdata.p_f]
        x_val2 = [x[0] / rdata.p_f[-1][0] for x in rdata.p_e]
        y_val2 = [x[1] for x in rdata.p_e]
        ax.scatter(x_val1, y_val1, linewidth=3)

        xarr = np.array(x_val1)
        yarr = np.array(y_val1)

        xpoly = poly.fit_transform(xarr.reshape(-1, 1))
        poly_reg_model.fit(xpoly, yarr)
        y_predicted = poly_reg_model.predict(xpoly)
        coef0 = poly_reg_model.intercept_
        coef1 = poly_reg_model.coef_
        coef = np.append(coef0, coef1)
        new_pol = np.polynomial.Polynomial(coef)
        new_pol_int = new_pol.integ()
        i_value = polyval(x=0.25, c=new_pol_int.coef)
        f_value = polyval(x=1, c=new_pol_int.coef)
        int_val = f_value - i_value
        if int_val < 0.15:
            col = colors[0]
        elif int_val < 0.3:
            col = colors[1]
        elif int_val < 0.5:
            col = colors[2]
        elif int_val >= 0.5:
            col = colors[3]
        # all_coef = np.append(all_coef, coef)

        plt.plot(xarr, y_predicted, color=col)
        # except:
        #     pass




ax.set_xlabel('time (s)', fontsize=16)
ax.set_ylabel(r'$p_e, p_f$', fontsize=20)
lgd = ax.legend([r'$p_f$', r'$p_e$'], fontsize=15, loc='upper left', frameon=False,
                bbox_to_anchor=(0.25, 1), ncol=2)
ax.set_title(r'Bayes estimate of $p_e$ and $p_f$', fontsize=16)
ax.set_ylim([0, 1.19])
ax.set_xlim([0, round(x_val1[-1] + 0.1)])
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.show()
