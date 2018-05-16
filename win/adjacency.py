import pandas as pd
from argparse import ArgumentParser
import utils
import networkx as nx
import os

arguments = ArgumentParser()
arguments.add_argument('--base', type=str, default="/scratch/balash/granger_output/")
arguments.add_argument('--input', type=str, default="all_lags_results_merged")
arguments.add_argument('--output', type=str, default="best_lags_strong_network")
args = arguments.parse_args()

df = utils.read_pickle(os.path.join(args.base, args.input))
G = nx.DiGraph()
for (w1, w2), stat in df.iteritems():
    if stat is None:
        continue
    G.add_node(w1)
    G.add_node(w2)
    if G.has_edge(w2, w1):
        G.remove_edge(w2, w1)
    else:
        G.add_edge(w1, w2, best_lag=stat)

utils.write_pickle(G, os.path.join(args.base, args.output))
shortest_paths = nx.all_pairs_shortest_path(G)
shortest_paths_file = args.output + "_shortest_paths"
utils.write_pickle(shortest_paths, os.path.join(args.base, shortest_paths_file))

            
