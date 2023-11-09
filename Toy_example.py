import infomap
import networkx as nx
import numpy as np
import pandas as pd
import numpy.linalg as linalg
from numpy.linalg import inv
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from itertools import product
from scipy.linalg import expm, sinm, cosm

# With this function, we create a small-world network with specified node-absorption configuration

def smallWorld(Nc,nc,kc,pc, DeltaC, Seed):
    
    # Defining empty graph
    G = nx.null_graph()
    
    # Appending Nc disconnected Watts-Strogatz subgraphs 
    for i in range(1,Nc+1):
        Gnew = nx.watts_strogatz_graph(nc,kc,pc)
        Gnew =  nx.convert_node_labels_to_integers(Gnew, first_label=nc*(i-1)+1, ordering='default')
        G = nx.compose(G,Gnew)
        
    # Setting edges between communities
    BetEdges = [(nc*i, nc*i+1) for i in range(1,Nc)] + [(Nc*nc,1)]                          
    G.add_edges_from(BetEdges)
    G = G.to_directed()


    # Adding node attributes to the final graph
    deltav =[]
    for I in range(1,Nc+1):
        for node in range(nc*(I-1)+1,nc*I+1):
            G.nodes[node]['abs'] = DeltaC[I-1]
            deltav.append(DeltaC[I-1])
    
    sizeAbs = [] 
    for node in G.nodes():
        if G.nodes[node]['abs'] == delta:
            sizeAbs.append(200)
        else:
            sizeAbs.append(500)
    # Adding edge weights
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    
    M0 = [k//nc for k in range(0,Nc*nc)]
    pos = nx.spring_layout(G,seed = Seed )
    #nx.draw(G,pos,node_color = deltav ,with_labels=True,cmap=plt.cm.Pastel1)
    nx.draw(G,pos,node_size=sizeAbs,with_labels=True, cmap = plt.cm.cool)
    plt.savefig('4commEx.eps', format = 'eps', dpi = 1000)
    return G, pos, deltav, M0, sizeAbs

# With this function, we create a grid network with a specified absorption configuration 

def gridexample(n, Deltav):
    G  = nx.grid_graph([2*n,2*n])
    sizeAbs =[]
    for node in G.nodes():
        i1 = node[0]
        i2 = node[1]
        
        if i1 in range(0,n) and i2 in range(0,n):
            G.nodes[node]['abs'] = Deltav[0]
            sizeAbs.append(200)
        if i1 in range(0,n) and i2 in range(n,2*n):
            G.nodes[node]['abs'] = Deltav[1]
            sizeAbs.append(600)
        if i1 in range(n,2*n) and i2 in range(0,n):
            G.nodes[node]['abs'] = Deltav[2]
            sizeAbs.append(900)
        if i1 in range(n,2*n) and i2 in range(n,2*n):
            G.nodes[node]['abs'] = Deltav[3]
            sizeAbs.append(1200)
    G = G.to_directed()
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    
    pos = {}
    deltavlist = []
    H = nx.convert_node_labels_to_integers(G,first_label=1)
    
    i = 1
    for node in G.nodes():
        pos[i] = node
        deltavlist.append(G.nodes[node]['abs'])
        i = i+1
    return H, deltavlist, pos, sizeAbs

# With this function, we plot graphs (also representing absorption and community structure)

def SingleComm(I, StructTem, StructNew, plotting, G, pos, tstar):
    
    idx = range(nc*(I-1),nc*I)
    nodes = range(nc*(I-1)+1, nc*I +1)
    Struct1 = [StructTem[int(i)] for i in idx]
    Struct2 = [StructNew[int(i)] for i in idx]
    
    if plotting == True:
        
        Gt = nx.watts_strogatz_graph(nc,kc,0)  
        Gt = nx.convert_node_labels_to_integers(Gt, first_label = nc*(I-1)+1)
        sizeAbs = [] 
        for node in Gt.nodes():
            if G.nodes[node]['abs'] == 0.2:
                sizeAbs.append(200)
            else:
                sizeAbs.append(700)
        nx.draw(Gt, node_size = sizeAbs, node_color = Struct1, with_labels = True, cmap = plt.cm.Pastel1)
        plt.show()

    else:    
        return metrics.adjusted_mutual_info_score(Struct1,Struct2)
    

# In this function we execute Algorithms 1

def infomap_abs(G,deltav, t, typeAdelta, h):
    
    # Define ajacency A and outdegree W
    
    AdeltaPos = True
    NumModules = []

    N = len(G.nodes)
    A = np.zeros((N,N))

    for edge in G.edges():
        i1 = int(edge[0])
        i2 = int(edge[1])  
        A[i2-1,i1-1] = 1
        
        
    outfl = A.sum(axis=0)
    W = np.diag(outfl)
    Ddelta = np.diag(deltav)
    
    #Defining the Laplacian 
    Ltilde = (W-A).dot(inv(h*W + Ddelta)) # use 'Linear', 'Exp'
    LtildeNor = (np.identity(N)-A.dot(inv(W))).dot(inv(Ddelta)) 
       
    # defining adjacency accounting for removal
    if typeAdelta == 'Linear': 
        Adelta = np.identity(N) - t*Ltilde
        tmax = np.multiply((deltav + h*outfl),1/outfl).min()
            
    if typeAdelta == 'Exp':
        Adelta = expm(-t*Ltilde)
        tmax = 100
        
    if typeAdelta == 'LinearNor': 
        Adelta = np.identity(N) - t*LtildeNor
        tmax = np.array(deltav).min()
            
    if typeAdelta == 'ExpNor':
        Adelta = expm(-t*LtildeNor)
        tmax = 100
    
    im = infomap.Infomap("--directed  --two-level --teleportation-probability 0")
    
    if typeAdelta == 'NormalInfo':
        Adelta = A
        im = infomap.Infomap("--directed " )
        tmax = 1
            
    for i in range(0,N):
        for j in range(0,N):
            im.add_link(i,j,Adelta[j,i])
            im.add_node(i,str(i+1))
            im.add_node(j,str(j+1))
    im.run()
    NumModules.append(im.numTopModules())

    Modv = pd.DataFrame(columns = ["node","Module"])
    ii = 0
    for node, module in im.modules:
        
        Modv.loc[ii,:] = np.array([node, module])
        ii = ii+1
   
    Modv = Modv.sort_values("node")
    True_cluster_info = list(Modv.loc[:,"Module"])
    return im.numTopModules(), True_cluster_info, tmax, Adelta 


def InfoAbs(G, deltav, t0, tf, Nt ,tstar, plotGraph, pos, sizeAbs, typeAdelta, h):
    
    if typeAdelta != 'NormalInfo':
        if t0>0 and tf>0:
            
            StructTem = [0]*len(list(G.nodes()))
            Nmodules = []
            Minfov = []
            MinfovComm = []
            tmax = infomap_abs(G, deltav, 1, typeAdelta, h)[2]
            if typeAdelta == 'Linear' or typeAdelta == 'LinearNor':
                print('Max. possible value of t: ', tmax)
            tend = np.array([tf,tmax]).min()
            th = (tend-t0)/Nt
            tv =np.arange(t0,tend + th,th)
            
    
            for t in tv:
                StructNew = infomap_abs(G, deltav, t, typeAdelta, h)[1]
                Nmodules.append(infomap_abs(G, deltav, t, typeAdelta, h)[0])
        
            Nmodules = np.array(Nmodules)

            plt.rcParams.update({'font.size': 14})

            
            plt.plot(tv,Nmodules)
            plt.xlabel(r'$t$')
            plt.ylabel('Number of communities')
            plt.savefig('ncomm_grid_9.eps', format='eps', dpi=1000)
            plt.show()

            

        if tstar != []:
            
            for t in tstar:
                
                M0inferred = list(infomap_abs(G, deltav, t, typeAdelta,h)[1])
                
                if plotGraph == True:

                    nx.draw(G,pos, node_size = sizeAbs, node_color = M0inferred, with_labels = True, cmap = plt.cm.tab20)
                    plt.savefig('StructGrid_9.eps', format = 'eps', dpi = 1000)
                    plt.show()
            
    else:
        M0inferred = list(infomap_abs(G, deltav, tstar, typeAdelta,h)[1])
        nx.draw(G,pos, node_size = sizeAbs, node_color = M0inferred, with_labels = True, cmap = plt.cm.Pastel1)
        plt.show()
        return np.array(infomap_abs(G, deltav, 1, typeAdelta, h)[1])


# We define a small-world network with 16 nodes 

Nc=4;nc = 4;kc = 4;pc = 0;seed = 13; delta = 1; Delta = 7
G = nx.null_graph()
G, pos, deltav, M0, sizeAbs = smallWorld(Nc, nc,kc,pc,[7,1,7,1], seed)

# We generate Figure 7  

InfoAbs(G,deltav,0.02,0.25, 100, [], True, pos,sizeAbs, 'Linear',0)
InfoAbs(G,deltav,0.4,10, 100, [], True, pos,sizeAbs, 'Exp',0)
InfoAbs(G,deltav,0.01,10, 100, [], True, pos,sizeAbs, 'Linear',1.5)
InfoAbs(G,deltav,0.4,20, 100, [], True, pos,sizeAbs, 'Exp',1.5)


# We define a grid graph with 36 nodes

n = 3
Deltaval = [0.2, 0.7, 1.2, 1.7]
G = gridexample(n,Deltaval)[0]
deltav = gridexample(n,Deltaval)[1]
pos = gridexample(n,Deltaval)[2]
sizeAbs = gridexample(n,Deltaval)[3]
nx.draw(G, pos, node_size = sizeAbs, node_color = [1]*len(list(G.nodes())), with_labels = True, cmap = plt.cm.cool)
plt.savefig('gridEx.eps',format= 'eps', dpi = 1000)

# We generate Figure 9 

InfoAbs(G,deltav,0.02,0.1, 100, [0.06], True, pos, sizeAbs, 'Exp',0)
InfoAbs(G,deltav,0.01,7, 100, [5.25], True, pos, sizeAbs, 'Exp',1)