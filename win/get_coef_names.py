import utils
import os
import numpy as np
import stattools
from argparse import ArgumentParser

ags = ArgumentParser()
ags.add_argument('--input', type=str)
ags.add_argument('--input2', type=str)
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

non_zeros = utils.read_pickle('{0}_{1}'.format(args.input2, args.index))
names = {}
for i,n  in non_zeros.iteritems():
    coeff_names = []
    for k, v in n[1].iteritems():
        coeff_names += [index[k]] 
    names[index[i]] = coeff_names
utils.write_pickle(names, '{0}_{1}'.format(args.output, args.index))
