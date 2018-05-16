import logging
import numpy
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool
import pickle
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def parse(path):
    df = pd.read_pickle(str(path))
    return df

arguments = ArgumentParser()
arguments.add_argument('--directory', type=Path)
args = arguments.parse_args()

df = parse(args.directory)
def get_all_words():
    # vocab_size = 100 #58221
    words = []
    for word1 in df.keys():
        for word2 in df.keys():
            if word1 == word2:
                continue
            words += [(word1, word2)]
    return words

with open('word-pairs', 'wb') as fp:
    pickle.dump(get_all_words(), fp)
