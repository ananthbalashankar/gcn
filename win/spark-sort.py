# pyspark --executor-memory 50G --driver-memory 20G --conf "spark.yarn.executor.memoryOverhead=12144" --conf  "spark.default.parallelism=100"
from statsmodels.tsa.stattools import grangercausalitytests

import sys
import pandas as pd
import numpy
from ast import literal_eval as make_tuple

def flatten(sparse_series):
    time_length = 800000
    time_series = [0 for i in range(time_length)]
    for (timestamp , tf) in sparse_series:
        try:
            time_series[timestamp] = tf
        except IndexError:
            print (timestamp, vocabid)
    sum_of_series = sum(time_series)
    time_series = [float(i)/sum_of_series for i in time_series]
    return time_series

def granger(val1, val2):
    lags = 10
    res = grangercausalitytests(numpy.transpose([flatten(val1), flatten(val2)]), 10, verbose=False)
    return (res[lags][0]['params_ftest'], res[lags][0]['lrtest'])

# Debug why reading is slow
# df_data = sc.textFile('/user/ab7325/causal-news' , 500)
# df_data = df_data.map(lambda a: make_tuple(a))
# df_keys = df_data.map(lambda (x, y): x)
# df_pairs = df_keys.cartesian(df_keys).filter(lambda (i, j): i != j)

# Debug why OOO
df = pd.read_pickle('causal-data')
df_keys = sc.parallelize(df.keys())
df_pairs = sc.parallelize([(i,j) for i in df.keys() for j in df.keys() if i != j])
df_pairs = df_pairs.repartition(1000)
df_data = df_keys.map(lambda x: (x, df[x]))
df_data = df_data.repartition(1000)
#df_map = sc.broadcast(df)
#results = df_pairs.map(lambda (x,y): ((x,y), granger(df_map.value[x], df_map.value[y])))
joined1 = df_pairs.join(df_data).map(lambda (i, (j, val)): (j, (i, val)))
joined2 = joined1.join(df_data).map(lambda (j, ((i, val1), val2)): ((i,j), (val1, val2)))
joined2 = joined2.repartition(10000)
results = joined2.mapValues(lambda (x,y): granger(x,y))
results.saveAsTextFile('/user/ab7325/causality-results-15')
