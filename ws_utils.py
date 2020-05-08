# ws_utils.py
# WESmith 05/08/20
# utilities for running pyphi experiments

import toolbox as tb
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 3)

def get_net(ledges, nodes, funcs, title=''):
    # wrapper for SP tools
    i, j = zip(*ledges)
    indexLUT = dict([(l,ord(l)-ord(nodes[0])) for l in sorted(set(i+j))])
    edges = [(indexLUT[i],indexLUT[j]) for i,j in ledges]
    net = tb.Net(edges=edges, title=title)
    for j,k in zip(nodes, funcs):
        net.get_node(j).func = k
    return net

def get_phi(perm, net):
    try:
        return net.phi(perm)
    except:
        return -1 # state isn't reachable

def run_expt(edges, nodes, funcs, title=''):
    net = get_net(edges, nodes, funcs, title=title)
    net.draw()
    df=net.tpm
    dd = [(k, get_phi(k, net)) for k in df.index]
    dff = pd.DataFrame(dd)
    dff.plot.hist(bins=100, title=title, grid=True)
