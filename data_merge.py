import pandas as pd
import pickle
import robot_data
def data_merge():
    titles = {'id': 17,
              'pre0': 18,
              'n01': 19, 'n02': 20, 'n03': 21, 'n04': 22, 'n05': 23, 'n06': 24,
              't01': 25, 't02': 26, 't03': 27, 't04': 28,
              'pre11': 30, 'pre12': 31,
              'n11': 32, 'n12': 33, 'n13': 34, 'n14': 35, 'n15': 36, 'n16': 37,
              't11': 38, 't12': 39, 't13': 40, 't14': 41,
              'ad11': 42, 'ad12': 43, 'ad13': 44, 'ad14': 45, 'ad15': 46, 'ad16': 47, 'ad17': 48, 'ad18': 49, 'ad19': 50,
              'ad110': 51, 'ad111': 52, 'ad112': 53, 'ad113': 54,
              'pre21': 56, 'pre22': 57,
              'n21': 58, 'n22': 59, 'n23': 60, 'n24': 61, 'n25': 62, 'n26': 63,
              't21': 64, 't22': 65, 't23': 66, 't24': 67,
              'ad21': 68, 'ad22': 69, 'ad23': 70, 'ad24': 71, 'ad25': 72, 'ad26': 73, 'ad27': 74, 'ad28': 75, 'ad29': 76,
              'ad210': 77, 'ad211': 78, 'ad212': 79, 'ad213': 80,
              'pre31': 82, 'pre32': 83,
              'n31': 84, 'n32': 85, 'n33': 86, 'n34': 87, 'n35': 88, 'n36': 89,
              't31': 90, 't32': 91, 't33': 92, 't34': 93,
              'ad31': 94, 'ad32': 95, 'ad33': 96, 'ad34': 97, 'ad35': 98, 'ad36': 99, 'ad37': 100, 'ad38': 101, 'ad39': 102,
              'ad310': 103, 'ad311': 104, 'ad312': 105, 'ad313': 106,
              'f1': 107, 'f2': 108, 'f3': 109, 'f4': 110, 'f5': 111,
              'u1': 112, 'u2': 113, 'u3': 114, 'u4': 115, 'u5': 116, 'u6': 117, 'u7': 118, 'u8': 119,
              'RT0': 120, 'RT1': 121, 'RT2': 122, 'RT3': 123,
              'oe': 124}
    ques = dict(zip(titles.values(), titles.keys()))
    df = pd.read_excel('expdata.xlsx')
    col1 = df.columns.values.tolist()
    col2 = list(range(len(col1)))
    col3 = dict(zip(col1, col2))
    df.rename(columns=col3, inplace=True)
    df.replace('Very much\n10', 10, inplace=True)
    df.replace('Not at all\n1', 1, inplace=True)
    df.replace('Very high\n10', 10, inplace=True)
    df[107].replace(['Strongly agree', 'Strongly disagree'], [7, 1], inplace=True)
    df[108].replace(['Strongly agree', 'Strongly disagree'], [7, 1], inplace=True)
    df[109].replace(['Strongly agree', 'Strongly disagree'], [7, 1], inplace=True)
    df[110].replace(['Strongly agree', 'Strongly disagree'], [7, 1], inplace=True)
    df[111].replace(['Strongly agree', 'Strongly disagree'], [7, 1], inplace=True)
    df.replace(['Strongly agree', 'Strongly disagree'], [5, 1], inplace=True)
    df.rename(columns=ques, inplace=True)
    df_experiment = df[ques.values()][1:]
    df_experiment.reset_index(inplace=True, drop=True)
    df_experiment['mode'] = ""
    df_experiment['T0_error'] = ""
    df_experiment['Rankings'] = ""

    for q in ['n04', 'n14', 'n24', 'n34']:
        for i, row in df_experiment.iterrows():
            df_experiment.at[i, q] = max(row[q], 100 - row[q])

    df2 = pd.read_excel('participants_id.xlsx')
    # print(df2.to_string())

    mv = {'(Mode 1) 3-4-2': 1,
          '(Mode 2) 3-2-4': 2,
          '(Mode 3) 4-3-2': 3,
          '(Mode 4) 4-2-3': 4,
          '(Mode 5) 2-3-4': 5,
          '(Mode 6) 2-4-3': 6}

    mv_t = {1: [3, 4, 2],
            2: [3, 2, 4],
            3: [4, 3, 2],
            4: [4, 2, 3],
            5: [2, 3, 4],
            6: [2, 4, 3]}
    for i in df_experiment['id']:
        mod = df2[df2['ID'] == int(i)]['Mode'].item()
        nerr0 = df2[df2['ID'] == int(i)]['Task 0'].item()
        ind = df_experiment.index[df_experiment['id'] == i].tolist()
        df_experiment.loc[ind[0], 'mode'] = mv[mod]
        df_experiment.loc[ind[0], 'T0_error'] = nerr0

        # ranking the tasks
        r1 = df_experiment.loc[ind[0], 'RT0']
        rt1 = 1
        r2 = df_experiment.loc[ind[0], 'RT1']
        rt2 = mv_t[mv[mod]][0]
        r3 = df_experiment.loc[ind[0], 'RT2']
        rt3 = mv_t[mv[mod]][1]
        r4 = df_experiment.loc[ind[0], 'RT3']
        rt4 = mv_t[mv[mod]][2]
        drank = [r1, r2, r3, r4]
        trank = [rt1, rt2, rt3, rt4]
        frank = dict(sorted(zip(drank, trank)))
        df_experiment.at[ind[0], 'Rankings'] = frank

    robot_data.robot_data(df_experiment)
    ope = df_experiment.pop('oe')
    df_experiment['oe'] = ope
    df_experiment.to_pickle("data1.pkl")

    # print(df_experiment.to_string())