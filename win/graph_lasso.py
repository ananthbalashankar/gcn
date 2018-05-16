import utils
import os
import numpy as np
import stattools_graph
from argparse import ArgumentParser

ags = ArgumentParser()
ags.add_argument('--input', type=str)
ags.add_argument('--alpha', type=float, default=1e-4)
ags.add_argument('--output', type=str, default='graph_lasso_matrix')
args = ags.parse_args()

ts = utils.read_pickle(args.input)
all_series = []
i = 0
index = {}
for k, v in ts.iteritems():
    all_series += [v]
    index[i] = k
    i+=1

all_series = np.transpose(all_series)

C = stattools_graph.grangercausalitytests(all_series, 10, alpha=args.alpha)

utils.write_pickle(C, os.path.join('/scratch/balash/granger_output', args.output))
