import math
import numpy as np
import pandas as pd
import numpy.linalg as linalg
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import inv
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from itertools import product
import networkx as nx
import random


### With this function, we create the small world network

def smallWorld(Nc,nc,kc,pc,Nb, delta, beta):

    # Defining empty graph
    G = nx.null_graph()
    
    # Appending Nc disconnected Watts-Strogatz subgraphs 

    for i in range(1,Nc+1):
        Gnew = nx.watts_strogatz_graph(nc,kc,pc)
        Gnew =  nx.convert_node_labels_to_integers(Gnew, first_label=nc*(i-1)+1, ordering='default')
        G = nx.compose(G,Gnew)
        
    # Setting edges between communities

    BetEdges = []
    BetNodes = []
    for j in range(1,Nb+1):
        pairC = list(np.random.choice(range(1,Nc+1),2,replace=False))     
        i1 = pairC[0]
        i2 = pairC[1]
        n1 = np.random.choice(range(nc*(i1-1)+1,nc*i1+1),1)[0]
        n2 = np.random.choice(range(nc*(i2-1)+1,nc*i2+1),1)[0]
        if (n1,n2) not in BetEdges and (n2,n1) not in BetEdges:
            BetEdges = BetEdges + [(n1,n2),(n2,n1)]
            BetNodes = BetNodes + [n1,n2]        
    G.add_edges_from(BetEdges)
    G = G.to_directed()

    # Defining a list with a label for each communities    
    M0 = np.array([k//nc for k in range(0,Nc*nc)])

    # Adding node attributes to the final graph
    for node in G.nodes():
        G.nodes[node]['abs'] = delta
        G.nodes[node]['state'] = 'S'
        G.nodes[node]['beta'] = beta 
    
    # Adding edge weights
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1/G.nodes[edge[0]]['abs']
        
    return G, BetEdges, list(set(BetNodes)), M0


### With the following functions, we set up transmission and absorption configurations (which are analogous to rewiring edges)

def rewiring0(G, BetEdges, BetNodes, delta, beta, Nrew):
    
    
    BetBalNodes = list(BetNodes) # X_0 nodes from which we draw bridging and balancing nodes
    
    BalNodes1 = [] # X_bal^1
    BetNodes1 = [] # X_bet^1
    
    BetNodesLeft = list(BetNodes) # X_0 \ X_bet^1  available nodes for picking linking nodes in the next stage
    BetEdgesLeft = list(BetEdges) # available edges for picking linking edges in the next stage
    
    for j in range(1,Nrew+1):
        
        # Picking a node between communities, with nodes i1 and i2
        edge0 = random.choice(BetEdges)        
        i1 = edge0[0]
        i2 = edge0[1]
        
        I1 = (i1-1)//nc +1 # community node 1        
        # nodes in the community of i1 that are not neighbors of i1 and are different from i1 
        comm1 = [x for x in set(range(nc*(I1-1)+1,nc*I1+1))-set(G.neighbors(i1))-set([i1]) if x in BetBalNodes]
        
        I2 = (i2-1)//nc +1 # community node 2
        # nodes in the community of i1 that are not neighbors of i1 and are different from i1 
        comm2 = [x for x in set(range(nc*(I2-1)+1,nc*I2+1))-set(G.neighbors(i2))-set([i2]) if x in BetBalNodes] 

        
        if (i1 in BetBalNodes) and (i2 in BetBalNodes) and len(comm1)>0 and len(comm2)>0:     
            
            # Removing linking nodes and edges
            BetEdgesLeft.remove((i1,i2))
            BetEdgesLeft.remove((i2,i1))                        
            BetNodesLeft.remove(i1)
            BetNodesLeft.remove(i2)
            
            # Picking Balancing nodes
            l1 = np.random.choice(comm1,1)[0]
            l2 = np.random.choice(comm2,1)[0]
            
            # Removing linking and balancing nodes from the drawing set
            BetBalNodes.remove(i1)
            BetBalNodes.remove(i2)
            BetBalNodes.remove(l1)
            BetBalNodes.remove(l2)
            
            # Adding nodes to construct X_bet^1 and X_bal^1
            BalNodes1 = BalNodes1 + [l1,l2]
            BetNodes1 = BetNodes1 + [i1,i2]
            
            
    return BalNodes1, BetNodes1, BetEdgesLeft, BetNodesLeft

# We use this function, balancing nodes can have large absoprtion

def rewiring1(G, BetEdges2, BetNodes2, delta, beta, Delta, Beta, Nrew):
    
    edge0 = random.choice(BetEdges2)
    i1 = edge0[0]
    i2 = edge0[1]
    
    l1 = -1
    l2 = -1
    
    if i1 in BetNodes2 and i2 in BetNodes2: 
        
        I1 = (i1-1)//nc +1
        comm1 = [x for x in set(range(nc*(I1-1)+1,nc*I1+1))-set([i1]) if x in BetNodes2]

        I2 = (i2-1)//nc +1
        comm2 = [x for x in set(range(nc*(I2-1)+1,nc*I2+1))-set([i2]) if x in BetNodes2] 

        if len(comm1)>0 and len(comm2)>0:  
            
            BetEdges2.remove((i1,i2))
            BetEdges2.remove((i2,i1))
            G.nodes[i1]['abs'] = Delta
            G.nodes[i2]['abs'] = Delta
            l1 = np.random.choice(comm1,1)[0]
            l2 = np.random.choice(comm2,1)[0]
            G.nodes[l1]['beta'] = Beta
            G.nodes[l2]['beta'] = Beta

            BetNodes2.remove(i1)
            BetNodes2.remove(i2)

            BetNodes2.remove(l1)
            BetNodes2.remove(l2)

        
    return G, BetEdges2, BetNodes2, [i1,i2], [l1,l2]

# With this function, we redefine absorption and transmission for bridging and balancing nodes, respectively

def BetaDelta(G, BalNodes1, BetNodes1, Beta, Delta):

    i1 = BetNodes1[0] # i_k
    i2 = BetNodes1[1] # i_{k+1}
    
    l1 = BalNodes1[0] # l_k
    l2 = BalNodes1[1] # l_{k+1}
    
    
    G.nodes[i1]['abs'] = Delta
    G.nodes[i2]['abs'] = Delta
    
    G.nodes[l1]['beta'] = Beta
    G.nodes[l2]['beta'] = Beta
    
    BalNodes1 = BalNodes1[2:]
    BetNodes1 = BetNodes1[2:]
    
    return G, BalNodes1, BetNodes1, [i1,i2], [l1, l2]


### With this function, we run a single SIR simulation. The parameter configurations are node attributes of G

def SIRgraph(G, findR0):    
    Ngen = 10
    N = len(G.nodes())
    Edges =  list(G.edges())
    Nodes = list(G.nodes())
    generation = np.zeros(N) ###
    #ColorV = np.zeros((N,1))
    Nr=0
    
    
    I0 = random.choice(Nodes) # Picking the first node labeled as I
    G.nodes[I0]['state'] = 'I'
    generation[I0-1] = 1  ###
    Ni = 1
    Niv = [1]
    
    t = 0 # Time
    Inodes = {}
    Inodes[I0] = G.nodes[I0]['abs']
    SIpairs={}
    

    for node in G.neighbors(I0):
        if G.nodes[node]['state'] == 'S':
            SIpairs[(node,I0)] = G.nodes[I0]['beta']

                
    while Ni > 0:
        distDelta = []
        distSIbeta = []
        for pair in SIpairs:
            distSIbeta = distSIbeta + [SIpairs[pair]]
        
        for node in Inodes:
            distDelta = distDelta + [Inodes[node]]
        
        LamSI = np.sum(np.array(distSIbeta))
        LamIDelta = np.sum(np.array(distDelta))       
        LamTot = LamSI + LamIDelta 
        
        pSI = list(np.array(distSIbeta)/LamSI)
        pDelta = list(np.array(distDelta)/LamIDelta)
        
        
        t = t + np.random.exponential((1/LamTot),1) # Picking the time of next event 

        u = random.uniform(0,1) # Picking a randon number to determine which event (infection or recovery) is occuring next

        if u < LamSI/LamTot: 
            
            SItn = np.random.choice(len(SIpairs),1,p=pSI) # Picking one SI pair
            SIt = list(SIpairs)[int(SItn[0])]
            
            Inew = SIt[0]
            Iold = SIt[1]
            
            G.nodes[Inew]['state'] = 'I'
            Inodes[Inew] = G.nodes[Inew]['abs']
            
            if findR0 == True:    
                generation[Inew-1] = generation[Iold-1] + 1   
            
                
            for node in G.neighbors(Inew): 
                
                if G.nodes[node]['state'] == 'I':
                    del SIpairs[(Inew,node)]

                if G.nodes[node]['state'] == 'S':
                    SIpairs[(node,Inew)] = G.nodes[Inew]['beta']
                    
            Ni = Ni+1

            
        if  u > LamSI/LamTot: 
            Irecn = np.random.choice(len(Inodes),1, p=pDelta)
            Irec = list(Inodes)[int(Irecn[0])] 
            
            del Inodes[Irec]
            G.nodes[Irec]['state'] = 'R'
    
                          
            for node in G.neighbors(Irec):
                if G.nodes[node]['state'] == 'S':
                    del SIpairs[(node,Irec)]
       
            Ni = Ni-1
            Nr = Nr+1

        Niv = Niv + [Ni]
    if findR0 == True:    
        R0dist = [len(np.where(np.array(generation)==i)[0]) for i in range(1,Ngen+1)]
    else:
        R0dist = []
    return t[0], max(Niv), Nr, R0dist


### With this function, we rewire and then run 'Nsim' SIR simulations (Algorithm 3 of the paper)

def FunctionalQuant(Nc,nc,kc,pc,Nb, Nsim, delta, Delta, beta, Nrew):

    Duration = []
    Peak = []
    FinalSize = []
    
    
    # Defining the starting graph
    
    G, BetEdges0, BetNodes0, M0 = smallWorld(Nc,nc,kc,pc,Nb, delta, beta)
    
    
    np.savetxt('SIR/EdgesG.csv',np.array(G.edges), delimiter = ',')
    np.savetxt('SIR/NodesG.csv',np.array(G.nodes), delimiter = ',')

    # Defining the nodes where delta and beta are changing
    
    BalNodes, BetNodes, BetEdges, BetNodes2tem =  rewiring0(G, BetEdges0, BetNodes0, delta, beta, Nrew)
    
    BetNodesAll = []
    BalNodesAll = []
    
    
    alpha = 0.1
    Beta = beta + alpha*delta*beta*(1/delta - 1/Delta)
    
    L0 = math.floor(len(BalNodes)/2)
    WhereR0v = [1, math.floor(L0)]
    
    BalNodes1 = list(BalNodes) # balancing nodes for first round
    BetNodes1 = list(BetNodes) # between nodes for first round
    
    BetEdges2 = []
    BetNodes2 = [] 
    
    for edge in BetEdges:
        if edge[0] in BetNodes2tem and edge[1] in BetNodes2tem:
            BetEdges2 = BetEdges2 + [edge]
            BetNodes2 = BetNodes2 + [edge[0],edge[1]]
    
    # first round
        
    for i in range(1, L0 + 1):
        

        DurTem = []
        PeakTem=[]
        FinalTem=[]
        R0distTem = []
        
        
        if i in WhereR0v:
            findR0 = True
        else:
            findR0 = False
     
        # SIR simulations 
        
        for j in range(0,Nsim):
            Dur,peak,Final, R0dist = SIRgraph(G,findR0)

            if Final > math.floor(0.02*Nc*nc):
                DurTem = DurTem + [Dur]
                PeakTem = PeakTem + [peak]
                FinalTem = FinalTem + [Final]

            if i in WhereR0v:
                R0distTem = R0distTem + [R0dist]
   
            for node in G.nodes():
                G.nodes[node]['state'] = 'S'


        if i in WhereR0v:
            np.savetxt('SIR/GenFirstRound'+str(int(i))+'SmallBeta.csv', np.array(R0distTem), delimiter=',')
            np.savetxt('SIR/Deltas'+str(int(i))+'FirstRound.csv', np.array(G.nodes.data('abs'))[:,1], delimiter = ',')

        Duration = Duration + [np.mean(np.array(DurTem))]
        Peak = Peak + [np.mean(np.array(PeakTem))]
        FinalSize = FinalSize + [np.mean(np.array(FinalTem))]
        
        G, BalNodes1, BetNodes1, Pair_bet, Pair_bal= BetaDelta(G, BalNodes1, BetNodes1, Beta, Delta)
        
        BetNodesAll = BetNodesAll + Pair_bet
        BalNodesAll = BalNodesAll + Pair_bal

        
    # Second round (we do it if we run out of potential balancing nodes with small absorption)
        
    Duration2 = []
    Peak2 = []
    FinalSize2 = []

    L0 = math.ceil(len(BetEdges2)/6)
    WhereR0v = [L0]

    for i in range(1,L0+1):
        if len(BetEdges2)>0:

            G, BetEdges2, BetNodes2, Pair_bet, Pair_bal = rewiring1(G, BetEdges2, BetNodes2, delta, beta, Delta, Beta, Nrew)
            BetNodesAll = BetNodesAll + Pair_bet
            BalNodesAll = BalNodesAll + Pair_bal
            
            DurTem = []
            PeakTem = []
            FinalTem = []
            R0distTem = []
            
            if i in WhereR0v:
                findR0 = True
            else:
                findR0 = False


            for j in range(0,Nsim):
                Dur,peak,Final, R0dist = SIRgraph(G,findR0)

                if Final > math.floor(0.02*Nc*nc):
                    DurTem = DurTem + [Dur]
                    PeakTem = PeakTem + [peak]
                    FinalTem = FinalTem + [Final]

                if i in WhereR0v:
                    R0distTem = R0distTem + [R0dist]

                for node in G.nodes():
                    G.nodes[node]['state'] = 'S'

        Duration2 = Duration2 + [np.mean(np.array(DurTem))]
        Peak2 = Peak2 + [np.mean(np.array(PeakTem))]
        FinalSize2 = FinalSize2 + [np.mean(np.array(FinalTem))]

        if i in WhereR0v:
            np.savetxt('GenSecondRound'+str(int(i))+'SmallBeta.csv', np.array(R0distTem), delimiter=',')
            np.savetxt('Deltas'+str(int(i))+'SecondRound.csv', np.array(G.nodes.data('abs'))[:,1], delimiter= ',')
    
    
    np.savetxt('SIR/QuantitiesBetaSmall.csv', (np.array(Duration+Duration2), np.array(Peak+Peak2), np.array(FinalSize+FinalSize2)), delimiter=',')
    np.savetxt('SIR/BetNodes.csv', np.array(BetNodesAll),delimiter = ',')
    np.savetxt('SIR/BalNodes.csv', np.array(BalNodesAll), delimiter = ',')

### Run Algorithm 3 of the paper to obtain the data for Figure 10 a--c

Nc = 20
nc = 12
kc = 6
pc = 0
Nb = 240
delta = 0.2
Delta  = 1
beta  =  0.125
Nsim = 1000
Nrew = 80

FunctionalQuant(Nc,nc,kc,pc,Nb, Nsim, delta, Delta, beta, Nrew)