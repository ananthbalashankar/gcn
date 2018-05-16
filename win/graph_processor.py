import pickle
import networkx as nx
import numpy as np
from scipy.cluster import hierarchy

def write_pickle(d, path):
    with open(path, 'wb') as fp:
        pickle.dump(d, fp)

def read_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

adj = read_pickle('/scratch/balash/granger_output/granger-reduced-filtered-adjacency')
G = nx.DiGraph()
for k, v in adj.iteritems():
    G.add_node(k)
    for i in v:
        G.add_edge(i, k)


        
# 1452
G.number_of_nodes()
# >70k
G.number_of_edges()

#In-out degree
out = A.in_degree().values()
s_out = np.sort(out)
counts, bin_edges = np.histogram(s_out, bins=20)
n_counts = counts/float(sum(counts))
cdf = np.cumsum(n_counts)
plt.plot(bin_edges[1:], cdf, label="In Degree", marker='^', color='r')
plt.savefig('/scratch/balash/final-output/in-out.png')

# number of strongly connected components = 1
G = utils.read_pickle('/scratch/balash/granger_output/lasso_alpha_5_network')
scc = list(nx.strongly_connected_components(G))
scc_lens = [len(i) for i in scc]
counts, bin_edges = np.histogram(scc_lens, bins=10)

def create_hc(G):
    """Creates hierarchical cluster of graph G from distance matrix"""
    path_length=nx.all_pairs_shortest_path_length(G)
    distances=np.zeros((len(G),len(G)))
    vocab = {}
    for u,p in path_length.items():
        for v,d in p.items():
            if u not in vocab:
                vocab[u] = len(vocab)
            if v not in vocab:
                vocab[v] = len(vocab)
            distances[vocab[u]][vocab[v]]=d
            
    return distances, vocab

distances, vocab = create_hc(G)
Z=hierarchy.complete(distances)
cl = hierarchy.fcluster(Z, t=75, criterion='maxclust')
counts, bin_edges = np.histogram(cl, bins=75)
cluster_ids = [j+1 for j,i in enumerate(counts) if i >= 5 and i <=30]

inv_vocab = {}
for k, v in vocab.iteritems():
    inv_vocab[v] = k

clusters = {}
for i in cluster_ids:
    cluster = set([])
    for k, v in enumerate(cl):
        if v == i:
            cluster.add(inv_vocab[k])
    clusters[i] = cluster

for k, v in clusters.iteritems():
     print v

utils.write_pickle(clusters, '/scratch/balash/granger_output/hierarchical_centroid_5_50_clusters')

# Takes a long time to run, still running
# number of cycles =
cycles = list(nx.simple_cycles(G))
len(cyles)

# All pair shortest paths
shortest_paths = nx.all_pairs_shortest_path(G)
write_pickle(shortest_paths, '/scratch/balash/granger_output/shortest_paths')

triangles = []
for i in ['monsoon']: #G.node:
     for j in G.node:
             for k in G.node:
                     if i!=j and j!=k and k!=i and G.has_edge(i,k) and G.has_edge(i,j) and G.has_edge(j,k):
                             triangles += [(i,j,k)]
write_pickle(triangles, '/scratch/balash/granger_output/triangles_in_the_network')

pairs = [('electricity', 'assembly'), ('electricity', 'atomic'), ('electricity', 'budget'), ('electricity', 'city'), ('electricity', 'coal'), ('electricity', 'heat'), ('land', 'arrest'), ('land', 'bill'), ('land', 'budget'), ('land', 'humidity'), ('land', 'climate'), ('land', 'force'), ('land', 'road'), ('monsoon', 'crop'), ('monsoon', 'flooding'), ('monsoon', 'rain'), ('monsoon', 'malaria'), ('price', 'crop'), ('price', 'festivities'), ('price', 'lakh'), ('price', 'land'), ('price', 'projects'), ('price', 'roads'), ('price', 'tariff'), ('rain', 'dam'), ('rain', 'flu'), ('rain', 'lakh'), ('rain', 'virus'), ('river', 'flu'), ('river', 'rain'), ('river', 'virus'), ('strike', 'attack'), ('strike', 'blast'), ('strike', 'protest'), ('strike', 'report'), ('strike', 'law')]

shortest_paths = utils.read_pickle('/scratch/balash/granger_output/best_lags_shortest_paths')
G = utils.read_pickle('/scratch/balash/granger_output/best_lags_network')
paths = {}
for (k,v) in  pairs:
     if k in shortest_paths:
             if v in shortest_paths[k]:
                     path = shortest_paths[k][v]
                     prev = None
                     path_with_lags = ""
                     for node in path:
                         if prev == None:
                             prev = node
                             path_with_lags += node
                             continue
                         path_with_lags += " --({0})-- {1}".format(G[prev][node]['best_lag'], node)
                         prev = node
                         
                     paths[(k,v)] = path_with_lags

for k, v in paths.iteritems():
    print k, v

shortest_paths = utils.read_pickle('/scratch/balash/granger_output/best_lags_shortest_paths')
vocab = {}
pairs = {}
for k,v in shortest_paths.iteritems():
    for w, p in v.iteritems():
        if k not in vocab:
            k_id = len(vocab) + 1
            vocab[k] = k_id
        if w not in vocab:
            w_id = len(vocab) + 1
            vocab[w] = w_id
        k_id = vocab[k]
        w_id = vocab[w]
        if (w_id,k_id) in pairs:
            pairs[(w_id,k_id)] = min(pairs[(w_id,k_id)], len(p))
        else:
            pairs[(k_id,w_id)] = len(p)

S = []
for (w1, w2), v in pairs.iteritems():
    S += [(w1, w2, v)]

utils.write_pickle(S, '/scratch/balash/granger_output/similarities')

import operator
clusters = utils.read_pickle('/scratch/balash/granger_output/spectral-clusters-min-12')
sinks = {}
sources = {}
for i, cluster in clusters.iteritems():
    H = G.subgraph(list(cluster))
    inD = H.in_degree()
    outD = H.out_degree()
    inOutRatio = {}
    for k, v in outD.iteritems():
        inOutRatio[k] = float(v)/(1 + inD[k])
    sortedRatios = sorted(inOutRatio.items(), key=operator.itemgetter(1))
    sinks[i] = sortedRatios[:5]
    sources[i] = sortedRatios[-5:]

for k, v in sinks.iteritems():
    print k, v

for k, v in sources.iteritems():
    print k, v
