import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial.polynomial import polyval


def creat_tasks(pattern_number):
    if pattern_number == 2:
        pattern = [
            ['orange', 'orange', 'green', 'green', 'pink'],
            ['pink', 'orange', 'orange', 'pink', 'blue'],
            ['pink', 'pink', 'blue', 'blue', 'green'],
            ['blue', 'green', 'green', 'blue', 'orange'],
            ['pink', 'pink', 'blue', 'green', 'orange']
        ]
    elif pattern_number == 3:
        pattern = [
            ['pink', 'orange', 'green', 'orange', 'pink'],
            ['green', 'pink', 'blue', 'pink', 'green'],
            ['blue', 'green', 'orange', 'green', 'blue'],
            ['orange', 'blue', 'pink', 'blue', 'orange'],
            ['pink', 'pink', 'blue', 'green', 'orange']
        ]
    elif pattern_number == 4:
        pattern = [
            ['orange', 'orange', 'blue', 'blue', 'pink'],
            ['blue', 'blue', 'pink', 'pink', 'green'],
            ['pink', 'pink', 'green', 'green', 'orange'],
            ['green', 'green', 'orange', 'orange', 'blue'],
            ['pink', 'pink', 'blue', 'green', 'orange']
        ]

    task_precedence_dict = {0: [], 1: [0], 2: [1], 3: [2], 4: [3],
                            5: [], 6: [5], 7: [6], 8: [7], 9: [8],
                            10: [], 11: [10], 12: [11], 13: [12], 14: [13],
                            15: [], 16: [15], 17: [16], 18: [17], 19: [18]}
    task_to_do = {}
    for j in task_precedence_dict:
        task_to_do[j] = {'workspace': j // 5 + 1, 'box': j % 5 + 1, 'color': pattern[j // 5][j % 5]}

    return task_to_do


def creat_table(data, pattern_num):
    def get_assigned_num(x):
        return int(str(x)[2:])

    def get_returned_num(x):
        for i in range(1, 6):
            if 3 * 10 ** i - x > 0:
                return x - 3 * 10 ** (i - 1)

    def get_human_colors(rob_colors):
        rob_colors_copy = [i for i in rob_colors]
        all_colors = ['green'] * 5 + ['blue'] * 5 + ['pink'] * 5 + ['orange'] * 5
        hcolor = []
        for item in all_colors:
            if item in rob_colors_copy:
                rob_colors_copy.remove(item)
            else:
                hcolor.append(item)
        return hcolor

    all_data = {}
    task_to_do = creat_tasks(pattern_num)
    wrong = [x[2] for x in data.action_times_human if (x[2] == 'Wrong_Return' or x[2] == 'Reject' or x[2] == 'Return')]
    wrong_corrected = [x[2] for x in data.action_times_human if
                       (x[2] == 'Correct_Return' or x[2] == 'Cancel_Wrong_Assign')]

    all_data['wrong'] = len(wrong)
    all_data['wrong_correct'] = len(wrong_corrected)
    hassign = [x[2] for x in data.action_times_human if x[2] == 'Assigned_to_Robot' or x[2] == 'Reject']
    hcassign = [x[2] for x in data.action_times_human if x[2] == 'Cancel_Assign' or x[2] == 'Cancel_Wrong_Assign']

    all_data['assign_by_human'] = len(hassign)
    all_data['canceled_by_human'] = len(hcassign)

    total_human_tasks = [x[2] for x in data.action_times_human if (x[2] == 'Wrong_Return' or
                                                                   x[2] == 'Return' or
                                                                   x[2] == 'Correct_Return' or
                                                                   x[2] == 'Human' or
                                                                   x[2] == 'Assigned_to_Human')]

    all_data['total_human_tasks'] = len(total_human_tasks)

    rassign = [x[3] for x in data.action_times_robot if x[5] == 'Assigned_to_Human']
    rassign_col = [task_to_do[get_assigned_num(x[6])]['color'] for x in data.action_times_robot if
                   x[5] == 'Assigned_to_Human']
    hassign_col = [task_to_do[x[6]]['color'] for x in data.action_times_robot if x[5] == 'Assigned_to_Robot']
    all_data['assign_by_robot'] = len(rassign)
    all_data['color_assign_by_robot'] = rassign_col
    all_data['color_assign_to_robot'] = hassign_col


    total_robot_tasks = [x[3] for x in data.action_times_robot if (x[5] == 'Robot' or
                                                                   x[5] == 'Assigned_to_Robot' or
                                                                   x[5] == 'Return' or
                                                                   x[5] == 'Human_by_Robot')]
    colors_by_robot = [task_to_do[x[6]]['color'] for x in data.action_times_robot if
                       (x[5] == 'Robot' or x[5] == 'Human_by_Robot' or x[5] == 'Assigned_to_Robot')]
    colors_by_human1 = [task_to_do[get_returned_num(x[6])]['color'] for x in data.action_times_robot if
                        (x[5] == 'Return')]
    colors_by_human2 = get_human_colors(colors_by_robot)
    colors_by_human = colors_by_human1 + colors_by_human2
    all_data['colors_done_human'] = colors_by_human
    all_data['colors_done_robot'] = colors_by_robot

    all_data['colors_done_human_blue'] = colors_by_human.count('blue')
    all_data['colors_done_human_pink'] = colors_by_human.count('pink')
    all_data['colors_done_human_orange'] = colors_by_human.count('orange')
    all_data['colors_done_human_green'] = colors_by_human.count('green')

    all_data['colors_done_robot_blue'] = colors_by_robot.count('blue')
    all_data['colors_done_robot_pink'] = colors_by_robot.count('pink')
    all_data['colors_done_robot_orange'] = colors_by_robot.count('orange')
    all_data['colors_done_robot_green'] = colors_by_robot.count('green')

    all_data['total_robot_tasks'] = len(total_robot_tasks)
    all_data['robot_travel_distance'] = sum(data.robot_travel_distance)
    all_data['human_travel_distance'] = sum(data.human_travel_distance)
    all_data['collaboration_time'] = data.experiment_end_time - data.experiment_start_time
    # data.dr = sum(data.robot_travel_distance)
    # data.dh = sum(data.human_travel_distance)
    # print('d total robot: ', data.dr)
    # print('d total human: ', data.dh)

    # print('time: ', data.experiment_end_time - data.experiment_start_time)
    return all_data


def get_pattern_number(df, task, id):
    modes = {1: [3, 4, 2],
             2: [3, 2, 4],
             3: [4, 3, 2],
             4: [4, 2, 3],
             5: [2, 3, 4],
             6: [2, 4, 3]}
    if task == 'task2.pickle':
        tn = 0
    elif task == 'task3.pickle':
        tn = 1
    elif task == 'task4.pickle':
        tn = 2
    row = df[df['id'] == id]
    mode = row['mode'].iloc[0]
    patter_number = modes[mode][tn]
    task_num = 'Task ' + str(tn + 1)
    return patter_number, task_num


def robot_data(df_experiment):
    df_experiment['Task 1'] = ''
    df_experiment['Task 2'] = ''
    df_experiment['Task 3'] = ''
    fig, ax = plt.subplots()
    poly = PolynomialFeatures(degree=3, include_bias=False)
    poly_reg_model = LinearRegression()
    colors = ['red', 'green', 'orange', 'yellow']

    ids = os.listdir('robot_data')
    all_int_vals = []
    for fol in ids:
        rec_tasks = os.listdir('robot_data/' + fol)
        if len(rec_tasks) < 3:
            print(fol, rec_tasks)
        if len(rec_tasks) == 3:
            for f in rec_tasks:
                file_name = 'robot_data/' + fol + '/' + f
                rfile = open(file_name, 'rb')
                rdata = pickle.load(rfile, encoding="latin1")
                pattern_num, task_num = get_pattern_number(df=df_experiment, task=f, id=fol)

                all_data = creat_table(rdata, pattern_num)
                x_val1 = [x[0] / rdata.p_f[-1][0] for x in rdata.p_f]
                y_val1 = [x[1] for x in rdata.p_f]
                x_val2 = [x[0] / rdata.p_f[-1][0] for x in rdata.p_e]
                y_val2 = [x[1] for x in rdata.p_e]
                # ax.scatter(x_val1, y_val1)

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
                i_value = polyval(x=0.2, c=new_pol_int.coef)
                f_value = polyval(x=1, c=new_pol_int.coef)
                int_val = f_value - i_value
                if int_val < 0.25:
                    col1 = colors[0]
                    # all_data['preference'] = 0
                elif int_val < 0.4:
                    col1 = colors[1]
                    # all_data['preference'] = 1
                elif int_val >= 0.4:
                    col1 = colors[3]
                    # all_data['preference'] = 2
                all_data['preference'] = int_val
                all_int_vals.append(int_val)

                df_experiment.loc[df_experiment['id'] == fol, task_num] = [all_data]
                plt.plot(xarr, y_predicted, color=col1)

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

    plt.hist(all_int_vals)
    plt.show()
    return df_experiment
