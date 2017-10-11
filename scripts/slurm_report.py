#!/usr/bin/python
import numpy as np
import time
import datetime as dt
import dateparser
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches

# input:
# sacct --allusers --starttime 2017-09-25 --format=User,Submit,start,end,nnodes > slurm_report.log
# cat slurm_report.log | awk '{print $1}' | sort -k1 | uniq
user_list = dict(zip(['xzhang1','huashli','afl','kayahan','nicola','yanmingw'], [[],[],[],[],[],[]]))

def s2t(tstring):
    i = dateparser.parse(tstring)
    if not i:
        raise ValueError('not able to parse %s' %tstring)
    return int(time.mktime(i.timetuple()))

job_list = []
with open("slurm_report.log", "r") as if_:
    lines = if_.readlines()[2:]
    for line in lines:
        l = line.split()
        if any([x=='Unknown' for x in l]) or (l[0] not in user_list):
            continue
        job_list_ = [l[0], s2t(l[1]), s2t(l[2]), s2t(l[3]), int(l[4])]  # user, submit, start, end, nnode
        job_list.append(job_list_)

def snapshot(t, job_list=job_list, user_list=user_list):    # t is int
    pending_job_list = [l for l in job_list if l[1] < t < l[2]]
    running_job_list = [l for l in job_list if l[2] < t < l[3]]
    # total nodes; should be 32, but if someone's job is blocking the way, that's all running job
    nnodes = sum([l[4] for l in running_job_list]) if pending_job_list else 32
    # running pie chart
    running_pie = {}
    for user in user_list:
        user_running_nnodes = sum([l[4] for l in running_job_list if l[0]==user]) if [l[4] for l in running_job_list if l[0]==user] else 0
        running_pie[user] = user_running_nnodes
    running_pie['free'] = nnodes - sum(running_pie.values())
    # overload
    running_nnodes = sum([l[4] for l in running_job_list])
    pending_nnodes = sum([l[4] for l in pending_job_list])
    # return
    return (running_pie, running_nnodes, pending_nnodes)

# some parsing
starttime = min([l[3] for l in job_list])
endtime = max([l[3] for l in job_list]) - 86400

# plotting
ts = np.arange(starttime, endtime, 3600)
tobjects = [dt.datetime.fromtimestamp(t) for t in ts]
running_nnodes_s = []
pending_nnodes_s = []
for t in ts:
    s = snapshot(t)
    for k in user_list:
        user_list[k].append(s[0][k])
    running_nnodes_s.append(s[1])
    pending_nnodes_s.append(s[2])

fig, ax = plt.subplots()
r = lambda: random.randint(0,255)
colors = ['#%02X%02X%02X' % (r(),r(),r()) for k in user_list]
ax.stackplot(tobjects, user_list.values(), colors=colors)
ax.plot(tobjects, pending_nnodes_s, color='k')
ax.legend([mpatches.Patch(color=c) for c in colors] + [mpatches.Patch(color='#000000')], user_list.keys() + ['pending'])
# plt.show()
plt.savefig('nanaimo_history.png', dpi=1500)
