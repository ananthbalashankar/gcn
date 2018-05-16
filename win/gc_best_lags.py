import logging
import numpy
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing.pool import ThreadPool
import pickle
import pandas as pd
from stattools import grangercausalitytests
from datetime import datetime
import utils

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

arguments = ArgumentParser()
arguments.add_argument('--path', type=str, default="flattened-timeseries")
arguments.add_argument('--index', type=int)
args = arguments.parse_args()

results = {}
lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 21, 30, 60 ,90]
df = utils.read_pickle(args.path)

def pairwise_granger(words):
    global df
    (word1, word2) = words
    if word1 not in df or word2 not in df:
        return (word1, word2, None)
    best_lag, res = grangercausalitytests(numpy.transpose([df[word2], df[word1]]), lags, verbose=False)
    causal_lag = None
    f_pvalue = res[0]['params_ftest'][1]
    lr_pvalue = res[0]['lrtest'][1]
    if f_pvalue < 0.01 and lr_pvalue < 0.01:
            causal_lag = best_lag 
    return (word1, word2, causal_lag)
    
pool = ThreadPool(processes=10)
with open ('pairs/word-pairs-{0}'.format(args.index), 'rb') as fp:
    all_words = pickle.load(fp)

for (w1, w2, res) in pool.imap_unordered(pairwise_granger, all_words):
    results[(w1, w2)] = res

utils.write_pickle(results, 'granger_output/aic_strong_results_{0}'.format(args.index))

    

