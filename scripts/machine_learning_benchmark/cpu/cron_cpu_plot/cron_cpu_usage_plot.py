import numpy as np
import matplotlib.pyplot as plt

with open("cron_cpu_usage.log", "r") as if_:
    l = np.float_([l.split() for l in if_.readlines()])
    plt.plot(l[:,0], l[:,1])
    plt.show()
