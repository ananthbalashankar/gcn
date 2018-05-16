from win import utils
import os
import numpy as np
from win import stattools
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
non_zeros = {}
for t in range(46):    
    all_series = []
    i = 0
    index = {}
    for k, v in ts.iteritems():
        if k.endswith('_{0}'.format(t)) == False:
            continue
        all_series += [v]
        index[i] = k
        i+=1
    max_i = len(all_series)
    all_series = np.transpose(all_series)
    for i in range(max_i):
        if i%args.max_index != args.index:
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
        error, coeff, best_vars = stattools.grangercausalitytests(all_series[:, cols], mxlg=6, alpha=args.alpha)
        top10_vars = dict(sorted(best_vars.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
        coeff_names = {}
        for k, v in coeff.iteritems():
            coeff_names[k] = index[k] 
        non_zeros[(t, i)] = (error, coeff, coeff_names, best_vars)
    
utils.write_pickle(non_zeros, '{0}/{0}_{1}'.format(args.output, args.index))

