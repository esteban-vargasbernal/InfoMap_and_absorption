import infomap
import networkx as nx
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import inv
from scipy.linalg import expm, sinm, cosm

### Loading the nodes, the edges and the absorption rates of the graph

Gnodes = np.loadtxt('NodesG.csv',delimiter = ',') 
Gedges = np.loadtxt('EdgesG.csv', delimiter = ',')
BetNodes = np.loadtxt('BetNodes.csv',delimiter = ',')
l = len(BetNodes)
BetNodes = BetNodes.reshape(int(l/2),2)


G = nx.null_graph()
G.add_nodes_from(list(Gnodes))
G.add_edges_from(list(Gedges))
G = G.to_directed()

for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1


### Getting the planted partition from standard InfoMap

N = len(G.nodes)
A = np.zeros((N,N))

for edge in G.edges():
    i1 = int(edge[0])
    i2 = int(edge[1])  
    A[i2-1,i1-1] = G[i1][i2]['weight']   

im = infomap.Infomap("--directed" )

for i in range(0,N):
    for j in range(0,N):
        im.add_link(i,j,A[j,i])
        im.add_node(i,str(i+1))
        im.add_node(j,str(j+1))
im.run()

natural_partition = im.get_modules()

### Computing the map function associated with an adaptation of InfoMap with exponential input for all parameter configurations

h = 0
t = 0.025

LM = []
for i in np.arange(1,69):
    
    deltav = np.loadtxt('deltas/stage_'+str(int(i))+'.csv', delimiter = ',')
        

    outfl = A.sum(axis=0)
    W = np.diag(outfl)
    Ddelta = np.diag(deltav)


    Ltilde = (W-A).dot(inv(h*W + Ddelta)) # use 'Linear', 'Exp'
    Adelta = expm(-t*Ltilde)


    im1 = infomap.Infomap("--directed  --two-level --teleportation-probability 0")

    for i in range(0,N):
        for j in range(0,N):
            im1.add_link(i,j,Adelta[j,i])
            im1.add_node(i,str(i+1))
            im1.add_node(j,str(j+1))

    im1.run(no_infomap = True, initial_partition = natural_partition)
    LM.append(im1.codelength)

# With this piece of code, we plot Figure 10 d

plt.rcParams.update({'font.size':15})
plt.plot(np.arange(1,69),LM,'k*')
plt.ylabel(r'$L(M_0)$', fontsize = 22)
plt.xlabel('Stage', fontsize = 22)
plt.tight_layout()
plt.savefig('Map_at_M0.eps',format = 'eps',dpi = 1000) 

