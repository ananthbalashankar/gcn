import utils
import os
import numpy as np
import stattools
import operator
from argparse import ArgumentParser

ags = ArgumentParser()
ags.add_argument('--input', type=str)
ags.add_argument('--alpha', type=float, default=5e-6)
ags.add_argument('--output', type=str)
ags.add_argument('--index', type=int)
ags.add_argument('--max_index', type=int, default=100)
args = ags.parse_args()

ts = utils.read_pickle(args.input)
all_series = []
i = 0
index = {}
stocks = set(['AAPL.csv','AMZN.csv','FB.csv','GOOG.csv','HPQ.csv','IBM.csv','MSFT.csv','ORCL.csv','stock-map','TSLA.csv','YHOO.csv'])
stock_i = []
for k, v in ts.iteritems():
    all_series += [v]
    index[i] = k
    if k in stocks:
        stock_i += [i]
    i+=1

all_series = np.transpose(all_series)

max_i = len(ts)
non_zeros = {}
for t, i in zip(range(len(stock_i)), stock_i):
    if t%args.max_index != args.index:
        continue
    perm = []
    for j in range(max_i):
        if j < i:
            perm += [j+1]
        elif j == i:
            perm += [0]
        else:
            perm += [j]
    cols = np.argsort(perm)
    error, coeff, best_vars = stattools.grangercausalitytests(all_series[:, cols], mxlg=30, alpha=args.alpha)
    top10_vars = dict(sorted(best_vars.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
    coeff_names = {}
    for k, v in coeff.iteritems():
        coeff_names[k] = index[k] 
    #    if k == 0:
    #        continue
    #    if k <= i:
    #        modified_coeff[k-1] = v
    #    else:
    #        modified_coeff[k] = v
    #non_zeros[i] = modified_coeff
    non_zeros[i] = (error, coeff, coeff_names, top10_vars)
    
utils.write_pickle(non_zeros, '{0}_{1}'.format(args.output, args.index))

def build_network(ts, non_zeros):
    import networkx as nx
    i = 0
    index = {}
    for k, v in ts.iteritems():
        index[i] = k
        i+=1
    G = nx.DiGraph()
    for x, v in non_zeros.iteritems():
        G.add_node(index[x])
        for y in v:
            G.add_node(index[y])
            G.add_edge(index[y], index[x])
    return G

#import utils
#non_zeros = utils.read_pickle('/scratch/balash/trigger-network_merged')
#ts = utils.read_pickle('/scratch/balash/trigger-stat')
#G = build_network(ts, non_zeros)
#utils.write_pickle(G, '/scratch/balash/final-output/trigger-nx-network')
