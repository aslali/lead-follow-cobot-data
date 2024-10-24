import pandas as pd
from data_merge import data_merge
import poststudy_data
import trust as tr
import confidence_helpfulness as ch
import confidence_trust as ct
import reliance as rl
import ueq
import nasatlx as tlx
import performance_time as pft
import performance_accuracy as pfa
import decitions_reilability as dr
import overall_performance as of
from robot_data import robot_data
import human_robot_analysis as hra
import fetch_human as fh
import perceived_performance as pp
import interview_preference as inp
renew_data = False
if renew_data:
    data_merge()

df_experiment = pd.read_pickle('data1.pkl')
# print(df_experiment.to_string())
poststudy_data.read_data()

r1 = []
r2 = []
r3 = []
r4 = []
for r in df_experiment['Rankings']:
    if type(r) is dict:
        r1.append(r[1])
        r2.append(r[2])
        r3.append(r[3])
        r4.append(r[4])
d1 = {i: r1.count(i) for i in r1}
print(d1)
d2 = {i: r2.count(i) for i in r2}
print(d2)
d3 = {i: r3.count(i) for i in r3}
print(d3)
d4 = {i: r4.count(i) for i in r4}
d4[4] = 0
print(d4)


rank_table = pd.DataFrame(data={'pattern': [1, 2, 3, 4], 'Most difficult': [25, 15, 8, 0], 'rank_2': [6, 11, 29, 2],
                                'rank_3': [8, 15, 7, 18], 'Lest difficult': [9, 7, 4, 28]})
# print(rank_table)
# print(df_experiment.to_string())

# tr.trust(df_experiment)
# tr.init_trust_other(df_experiment)

# ch.conf_prehelp_all(df_experiment)
# ch.prehelp_by_pattern(df_experiment)
# ch.prehelp_by_task(df_experiment)
# ct.self_confidence(df_experiment)
# ct.prepare_trust_data(df_experiment)
# rl.posthelp(df_experiment)
# pft.performance_time(df_experiment)
# pfa.performance_accuracy(df_experiment)
# pp.performance(df_experiment)
# dr.decisions_reliability(df_experiment)
# ueq.plot_ueq(df_experiment)

# of.plot_overall_perf(df_experiment)


# tlx.tlx_mental(df_experiment)
# tlx.tlx_physical(df_experiment)
# tlx.tlx_temporal(df_experiment)
# tlx.tlx_performance(df_experiment)
# tlx.tlx_effort(df_experiment)
# tlx.tlx_frust(df_experiment)

# robot_data(df_experiment)

# hra.lead_follow_by_task_pattern_rank(df_experiment)
# hra.lead_follow_by_mode(df_experiment)
# hra.lead_follow_mean(df_experiment)
# hra.lead_follow_human_assigned(df_experiment)
# hra.lead_follow_robot_assigned((df_experiment))
# hra.lead_follow_wrong(df_experiment)
# hra.lead_follow_distance(df_experiment)
# hra.lead_follow_ntask(df_experiment)
# hra.lead_follow_blue(df_experiment)
# hra.lead_follow_orange(df_experiment)
# hra.lead_follow_pink_green(df_experiment)
# hra.lead_follow_time(df_experiment)
hra.plot_correlations(df_experiment)
##hra.robot_human_colors(df_experiment)
# hra.robot_human_tasks_done_pattern(df_experiment)
# poststudy_data.preference_style(df_experiment)
# poststudy_data.interview_poststudy(df_experiment)
# fh.fetch_human(df_experiment)

# inp.interview_lead_follow2(df_experiment)