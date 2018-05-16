import pandas as pd
from pathlib import Path

def calculateTotalOccurrences(path):
    df = pd.read_pickle(Path(path))
    return sum(df)

def readSeries(path):
    return pd.read_pickle(Path(path))


def timeSeriesToPickle():
    valid_words = pd.read_pickle('valid_words')
    time_series = {}
    for i, r in valid_words.iterrows():
        if (i%10) == 0:
            print "Completed {0} words out of 11241".format(i)
            time_series[r[0]] = readSeries('/home/balash/scratch2/sunandan/time_series/all/{0}.pandas'.format(r[0]))
    time_series = pd.DataFrame(time_series)
    time_series.to_pickle('time_series')


def stemWords():
    from operator import add
    import utils
    from nltk.stem.porter import *
    stemmer = PorterStemmer()
    D2 = utils.read_pickle('../all-word-series')
    D3 = {}
    for k, v in D2.iteritems():
        sk = stemmer.stem(k)
        if sk not in D3:
            D3[sk] = v
        else:
            D3[sk] = map(add, D3[sk], v)
        
    utils.write_pickle(D3, '../stemmed-all-unigrams')
    
    
import utils
import numpy as np
from statsmodels.tsa.stattools import adfuller

from argparse import ArgumentParser

ags = ArgumentParser()
ags.add_argument('--input', type=str)
ags.add_argument('--output', type=str)
args = ags.parse_args()
ts = utils.read_pickle(args.input)
stat = set([])
diff_stat = set([])
non_stat = set([])
stat_series = {}
for k, v in ts.iteritems():
    if (adfuller(v)[1] < 0.05):
        stat.add(k)
        stat_series[k] = v
    elif adfuller(np.diff(v))[1] < 0.05:
        diff_stat.add(k)
        stat_series[k] = np.concatenate(([0], np.diff(v)))
    else:
        non_stat.add(k)

utils.write_pickle(stat_series, args.output)
