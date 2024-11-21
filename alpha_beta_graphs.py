

from sys import set_asyncgen_hooks
import pandas as pd
import networkx as nx
from earlyadopters import get_early_adopters
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import set_epoch

# this script generates alpha - beta graphs per forum

# forum id is specified below, as well as our alpha and beta (alpha*multiplier)
forums = [84]#[34,41,77,84]
alpha = 20
m = 3

for f in forums:
    # run query to retrieve topics id's where above > size and > size/2
    csc, ncsc, tcsc, tncsc = get_early_adopters(f, alpha, alpha*m) # set threshold here    

    topic = []
    start = []
    end = []
    end_beta = []

    for key, users in csc.items():
        topic.append(key)
        start.append(tcsc[key][0])
        end.append(tcsc[key][alpha-1])
        end_beta.append(tcsc[key][-1])

    topic_np = np.array(topic)
    start_np = np.array(start)
    end_np = np.array(end)
    end_np_beta = np.array(end_beta)

    start_sort = np.sort(start_np)
    end_sort = end_np[np.argsort(start_np)]
    end_sort_beta = end_np_beta[np.argsort(start_np)]
    topic_sort = topic_np[np.argsort(start_np)]

    mdates = set_epoch(start_sort[0])

    plt.barh(range(len(start_sort)), end_sort_beta-start_sort, left=start_sort, align='center', color = 'red')
    plt.barh(range(len(start_sort)), end_sort-start_sort, left=start_sort, align='center')

    plt.yticks(range(len(start_sort)), topic_sort)
    plt.xlabel('Days Passed since first post by root user')
    plt.title(f'Forum {f} with alpha {alpha} and multiplier {m}X')

    print(f'\n\nThis is forum {f} with alpha {alpha} and multiplier {m}X')
    plt.show()
