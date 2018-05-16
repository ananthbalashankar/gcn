import utils
import os
import numpy as np
import stattools
from argparse import ArgumentParser

ags = ArgumentParser()
ags.add_argument('--base', type=str, default='/scratch/balash/granger_output')
ags.add_argument('--merged', type=str)
ags.add_argument('--stat', type=str)
ags.add_argument('--output', type=str)
args = ags.parse_args()

def build_network(ts, non_zeros):
    import networkx as nx
    i = 0
    index = {}
    for k, v in ts.iteritems():
        index[i] = k
        i+=1
    G = nx.DiGraph()
    for x, v in non_zeros.iteritems():
        G.add_node(index[x])
        for y, bestlag in v.iteritems():
            G.add_node(index[y])
            G.add_edge(index[y], index[x], lag=bestlag)
    return G

non_zeros = utils.read_pickle(os.path.join(args.base, args.merged))
ts = utils.read_pickle(os.path.join(args.base, args.stat))
G = build_network(ts, non_zeros)
utils.write_pickle(G, os.path.join(args.base, args.output))

def build_lag_network(G, minlag, maxlag):
    import networkx as nx
    L = nx.DiGraph()
    for (u, v, l) in G.edges_iter(data='lag'):
        if l <= maxlag and l>=minlag:
            L.add_node(u)
            L.add_node(v)
            L.add_edge(u,v,lag=l)
    return L

def topic_subgraphs(G, inv_top):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    H = []
    for i in range(100):
        stemmed = [stemmer.stem(j) for j in inv_top[i]] 
        F = G.subgraph(stemmed)
        H.append(F)
        print (i, F.number_of_nodes(), F.number_of_edges())
    return H

def word2topic():
    from gensim import models, corpora
    lda = models.LdaModel.load('/scratch/sunandan/TOI_LDA/toi.lda')
    d = corpora.Dictionary.load('/scratch/balash/toi.dict')
    topic_terms = {}
    for k, v in d.iteritems():
        topics = lda.get_term_topics(k, minimum_probability=1e-3)
        topic_terms[v] = topics
        inv_top = {}
        term2top = {}
    for i in range(100):
        inv_top[i] = set([])
    for k, v in topic_terms.iteritems():
        if k not in term2top:
            term2top[k] = set([])
        for (t, p) in v:
            inv_top[t].add(k)
            term2top[k].add(t)
    return term2top, inv_top

term2top, top2term = word2topic()

A = utils.read_pickle('/scratch/balash/final-output/all-words-in-topics-graph')
topicGraph = nx.DiGraph()
for i in range(100):
    topicGraph.add_node(i)

for (u, v) in A.edges_iter():
    if bool(term2top[u].intersection(term2top[v])) == False:
        for i in term2top[u]:
            for j in term2top[v]:
                if topicGraph.has_edge(i, j):
                    topicGraph[i][j]['weight'] += 1.0
                else:
                    topicGraph.add_edge(i,j, weight=1.0)

top2termcount = {}
for i in range(100):
     top2termcount[i] = len(top2term[i])

for (u,v,w) in topicGraph.edges_iter(data='weight'):
     topicGraph[u][v]['weight'] = (topicGraph[u][v]['weight']/(top2termcount[u]*top2termcount[v]))*100.0

weights = []
for (u,v,w) in topicGraph.edges_iter(data='weight'):
    weights += [w]

import numpy as np
np.median(weights)
def prune(topicGraph, thres):
    fTopicGraph = nx.DiGraph(topicGraph)
    for (u,v,w) in fTopicGraph.edges_iter(data='weight'):
        if w < thres:
            fTopicGraph.remove_edge(u,v)
    return fTopicGraph

thres = np.percentile(weights, 99.9)
fTopicGraph = prune(topicGraph, thres)
F = nx.DiGraph(A)
for (u,v) in A.edges_iter():
    has_edge = False
    if bool(term2top[u].intersection(term2top[v])):
        F.remove_edge(u,v)
        continue
    for i in term2top[u]:
        for j in term2top[v]:
            if fTopicGraph.has_edge(i, j):
                has_edge = True
    if has_edge == False:
        F.remove_edge(u,v)

nx.write_graphml(F, '/scratch/balash/pruned-graph.graphml')

inv_top = word2topic()
G = utils.read_pickle('/scratch/balash/final-output/lasso/lasso_alpha_5_6_network')
H = topic_subgraphs(G, inv_top)
utils.write_pickle(H, '/scratch/balash/final-output/topic_subgraphs')

#Interesting subgraphs
for i in range(100):
    if W[i].number_of_nodes() < 40 and W[i].number_of_edges()>5:
        print (i, W[i].number_of_nodes(), W[i].number_of_edges())


import networkx as nx

from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models.graphs import from_networkx
A = utils.read_pickle('/scratch/balash/final-output/all-words-in-topics-graph')

#Bokeh
plot = figure(title="Word graph with topic coloring", x_range=(-1.1,1.1), y_range=(-1.1,1.1), tools="", toolbar_location=None)
graph = from_networkx(A, nx.spring_layout, scale=10, center=(0,0))
plot.renderers.append(graph)
output_file("/scratch/balash/final-output/word-bokeh.html")
show(plot)

#Matplotlib
colors = [0]*A.number_of_nodes()
nodes = G.nodes()
#C = ['green', 'blue', 'yellow', 'red', 'orange']
#clusters = [44, 59, 73, 85, 22]
m = 0
import matplotlib.pyplot as plt

pos= {}
r=20
import math
import random
for i in range(100):
    x = [nodes.index(j) for j in H[i].nodes()]
    for t in x:
        colors[t] = i
        xrand = random.random()
        yrand = random.random()
        pos[nodes[t]] = (r*math.cos((i/100.0)*(2*math.pi)) + xrand, (r*math.sin((i/100.0)*(2*math.pi)) + yrand))

plt.clf()
nx.draw_networkx(F, pos=pos, node_size=10, nodelist=nodes, node_color=colors, cmap=plt.cm.prism, vmin=0, vmax=100, edge_color='lightgrey', width=0.1, with_labels=False)
plt.savefig('/scratch/balash/final-output/grouped-words.png', dpi=1000)


def find_word(G, w):
    bigrams = []
    for x in G.nodes():
        if type(x) == tuple and (x[0] == w or x[1] == w):
            bigrams += [x]
    return bigrams

word_pairs = [('price', 'projects'), ('strike', 'attack'), ('river', 'rain'), ('land', 'budget'), ('electricity', 'city'), ('price', 'land'), ('strike', 'law'), ('land', 'bill'), ('land', 'road'), ('monsoon', 'rain'), ('strike', 'protest'), ('strike', 'report'), ('price', 'roads'), ('rain', 'lakh'), ('price', 'lakh'), ('land', 'arrest'), ('land', 'force')]

import networkx as nx
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
N_L = []
N_pos = []
for (x, y) in word_pairs:
    N = nx.DiGraph()
    N.add_node(x)
    N.add_node(y)
    stem_y = stemmer.stem(y)
    stem_x = stemmer.stem(x)
    v = [stem_y] + find_word(G, stem_y)
    u = [stem_x] + find_word(G, stem_x)
    e = 0
    pos = {}
    for i in u:
        for j in v:
             if G.has_edge(i, j):
                 print (i, j, G[i][j]['lag'])
                 N.add_edge(x, i)
                 N.add_edge(i, j, lag=G[i][j]['lag'])
                 N.add_edge(j, y)
                 pos[i] = (2, e)
                 pos[j] = (10, e)
                 e+=1
    pos[x] = (0, e/2)
    pos[y] = (12, e/2)
    N_L += [N]
    N_pos += [pos]

#import matplotlib.pyplot as plt
m = 0
for H in N_L:
    plt.clf()
    nx.draw_networkx(H, pos=N_pos[m], node_size=10, node_color='blue', edge_color='grey', width=1, with_labels=True)
    plt.savefig('/scratch/balash/final-output/causal_pairs_{0}.png'.format(m), dpi=1000)
    m += 1
    



(10, 26, 11)
(15, 23, 21)
(17, 18, 10)
(22, 36, 50)
(24, 35, 21)
(32, 32, 28)
(36, 19, 22)
(38, 31, 46)
(39, 23, 15)
(41, 18, 6)
(42, 14, 7)
(44, 17, 9)
(47, 19, 14)
(50, 32, 31)
(53, 30, 31)
(58, 39, 34)
(59, 19, 11)
(60, 21, 15)
(61, 17, 14)
(64, 38, 30)
(65, 24, 30)
(69, 25, 22)
(73, 36, 23)
(83, 10, 6)
(85, 19, 6)
(87, 22, 15)
(88, 12, 6)
(90, 32, 33)
(96, 30, 14)
        
# H[44].edges()
[(u'sister', u'khan'), (u'mumbai', u'sanjay'), (u'mumbai', u'khan'), (u'brother', u'close'), (u'brother', u'actor'), (u'khan', u'close'), (u'patil', u'sister'), (u'patil', u'sanjay'), (u'friend', u'shah')]

W[59].edges()
[(u'district', u'rural'), (u'area', u'start'), (u'raj', u'visit'), (u'raj', u'panchayat'), (u'field', u'panchayat'), (u'field', u'raj'), (u'field', u'bound'), (u'visit', u'home'), (u'visit', u'work'), (u'local', u'bound'), (u'panchayat', u'bound')]

W[73].edges()
[(u'set', u'track'), (u'set', u'young'), (u'crowd', u'host'), (u'crowd', u'indian'), (u'crowd', u'began'), (u'crowd', u'young'), (u'crowd', u'stage'), (u'ravi', u'live'), (u'ravi', u'art'), (u'style', u'western'), (u'style', u'form'), (u'perform', u'duo'), (u'began', u'duo'), (u'indian', u'art'), (u'high', u'track'), (u'high', u'hear'), (u'great', u'host'), (u'time', u'music'), (u'time', u'young'), (u'popular', u'perform'), (u'hear', u'record'), (u'hear', u'music'), (u'hear', u'duo')]

W[85].edges()
[(u'power', u'review'), (u'phase', u'impact'), (u'phase', u'run'), (u'clean', u'partner'), (u'clean', u'air'), (u'action', u'impact')]
    
