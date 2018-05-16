import pickle
import pandas as pd
from datetime import datetime


from argparse import ArgumentParser

ags = ArgumentParser()
ags.add_argument('--input', type=str)
ags.add_argument('--output', type=str)
args = ags.parse_args()

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


def day_of_year(d1):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    year = d1.year
    day = int(d1.strftime("%j"))
    if day > 59 and (year % 4 == 0):
        return day - 2
    return day - 1


def day_in_year(d):
    return d % 365

from os import listdir
from os.path import isfile, join
import pandas as pd

def parse(path):
    flattened_df = {}
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for p in onlyfiles:
        try:
            with open(join(path, p), "rb") as fp:
                df = pd.read_csv(fp, sep='\t', lineterminator='\n', header=None)
                series = [0] * 365
            for _, t in df.iterrows():
                if t[0] == "2008-02-29" or t[0] == "2012-02-29":
                    continue
                try:
                    series[day_of_year(t[0])] += t[1]
                except:
                    print (t[0], p)
                    continue
            sum_of_series = sum(series)
            if sum_of_series > 0:
                series = [float(i)/sum_of_series for i in series]
            flattened_df[p] = series
        except:
            print ('Not processed: {0}'.format(p))
    with open(args.output, "wb") as fp:
        pickle.dump(flattened_df, fp)


parse(args.input)
