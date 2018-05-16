import logging
import numpy
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing.pool import ThreadPool
import pickle
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import datetime

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

arguments = ArgumentParser()
arguments.add_argument('--path', type=str, default="all-word-pairs")
arguments.add_argument('--lags', type=int, default=10)
arguments.add_argument('--index', type=int)
args = arguments.parse_args()

results = {}
lags = args.lags
with open(args.path, "rb") as fp:
    df = pickle.load(fp)

def pairwise_granger(words):
    global df
    (word1, word2) = words
    res = grangercausalitytests(numpy.transpose([df[word1], df[word2]]), lags, verbose=False)
    return (word1, word2,  [res[lags][0]['params_ftest'], res[lags][0]['lrtest']])
    
pool = ThreadPool(processes=25)
with open ('pairs/word-pairs-{0}'.format(args.index), 'rb') as fp:
    all_words = pickle.load(fp)

for (w1, w2, res) in pool.imap_unordered(pairwise_granger, all_words):
    results[(w1, w2)] = res
output_df = pd.DataFrame(results)
output_df.to_pickle('granger_output/granger_results-{0}'.format(args.index))

    

