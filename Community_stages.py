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

Deltav1 = np.loadtxt('Deltas1FirstRound.csv', delimiter = ',')
Deltav2 = np.loadtxt('Deltas29FirstRound.csv', delimiter = ',')
Deltav3 = np.loadtxt('Deltas39SecondRound.csv', delimiter = ',')

G = nx.null_graph()
G.add_nodes_from(list(Gnodes))
G.add_edges_from(list(Gedges))
G = G.to_directed()

for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1


# With this function, we can plot a given partition for a specific planted community 
# Examples ot the output of this function are Figures 11c and 11d for the 
# fifth planted community and the induced partition obtained from Algorithm 1b

def SingleComm(I, StructTem, StructNew, plotting, G, tstar, deltav):
    
    idx = range(nc*(I-1),nc*I)
    nodes = range(nc*(I-1)+1, nc*I +1)
    Struct1 = [StructTem[int(i)] for i in idx]
    Struct2 = [StructNew[int(i)] for i in idx]
        
    
    if plotting == True:
    
        Gt = nx.watts_strogatz_graph(nc,6,0)
        Gt = nx.convert_node_labels_to_integers(Gt,first_label=nc*(I-1)+1)
        sizeAbs = [] 
        for i in idx:
            if deltav[i] == 0.2:
                sizeAbs.append(400)
            else:
                sizeAbs.append(750)
        nx.draw(Gt, poss, node_size= sizeAbs,  node_color = Struct1, with_labels = True, cmap = plt.cm.tab10)
        plt.savefig('singlecomm'+str(I)+'deltav3'+str(tstar)+'.eps',format='eps',dpi =1000)
        plt.show()
        return Struct1

    else:
        return Struct1


# With this function, we can run Algorithms 1a and Algorithms 1b for specified 
# Markov time t and specified H = h I

def infomap_abs(G,deltav, t, typeAdelta, h):
        
    AdeltaPos = True
    NumModules = []
    MutualInfo = []

    N = len(G.nodes)
    A = np.zeros((N,N))

    for edge in G.edges():
        i1 = int(edge[0])
        i2 = int(edge[1])  
        A[i2-1,i1-1] = G[i1][i2]['weight']   
        
    outfl = A.sum(axis=0)
    W = np.diag(outfl)
    Ddelta = np.diag(deltav)
    
    # Defining the Laplacian in step 1 of Algorithms 1a and 1b
    
    Ltilde = (W-A).dot(inv(h*W + Ddelta)) # use 'Linear', 'Exp'
    LtildeNor = (np.identity(N)-A.dot(inv(W))).dot(inv(Ddelta)) 
       
    # Defining the transition matrix P_l or P_e that will be input of regular InfoMap 
    
    if typeAdelta == 'Linear': 
        Adelta = np.identity(N) - t*Ltilde
        tmax = np.multiply((deltav + h*outfl),1/outfl).min()
            
    if typeAdelta == 'Exp':
        Adelta = expm(-t*Ltilde)
        tmax = 10
        
    if typeAdelta == 'LinearNor': 
        Adelta = np.identity(N) - t*LtildeNor
        tmax = deltav.min()
            
    if typeAdelta == 'ExpNor':
        Adelta = expm(-t*LtildeNor)
        tmax = 10
    
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

        
    Modv = pd.DataFrame(Modv, columns = ["node","Module"])
    Modv = Modv.sort_values("node")
    True_cluster_info = list(Modv.loc[:,"Module"])
    return im.numTopModules(), True_cluster_info, tmax 

# With this function, we can run Algorithm 1a or 1b for Nt equally spaced Markov times 
# in the interval [t0,tf]. In addition, with this function we can plot Figure 11 

def InfoAbs(G, deltav, t0, tf, Nt ,tstarv, typeAdelta, h, M0, Istarv): 
    if typeAdelta != 'NormalInfo':
        if t0>0 and tf>0:
            StructTem = M0
            Nmodules = []
            Minfov = []
            MinfovComm = []
            tmax = infomap_abs(G, deltav, 1, typeAdelta, h)[2]
            if typeAdelta == 'Linear' or typeAdelta == 'LinearNor':
                print('Max. possible value of t: ', tmax)
            tend = np.array([tf,tmax]).min()
            th = (tend-t0)/Nt
            tv =np.arange(t0,tend + th,th)
            ncommdf = pd.DataFrame(index = tv, columns = ['comm ' + str(i) for i in range(1,21)])

            for t in tv:
                Nmodules.append(infomap_abs(G, deltav, t, typeAdelta, h)[0])
                StructNew = list(infomap_abs(G, deltav, t, typeAdelta,h)[1])
                StructTem = list(StructNew)
                comm = np.array(StructNew).reshape(20,12)
                ncomm = []
                for i in range(0,20):
                    ncomm = ncomm + [len(set(list(comm[i,:])))]
                ncomm = np.array(ncomm)
                ncommdf.loc[t,:] = ncomm
                
                
            Nmodules = np.array(Nmodules)    
            plt.plot(tv,Nmodules)
            plt.xlabel(r'$t$')
            plt.ylabel('Number of communities')
            plt.show()    
            

        if tstarv != []: 
            
            ncomm = []
            for i in range(0,20):
                ncomm = ncomm + [len(set(list(comm[i,:])))]
            ncomm = np.array(ncomm)

            ncomm = np.array(ncommdf['comm '+str(5)])
  
            for Istar in Istarv:
                for tstar in tstarv:
                    M0inferred = list(infomap_abs(G, deltav, tstar, typeAdelta,h)[1])
                    print('I = ',Istar,', t =',tstar)
                    SingleComm(Istar, M0inferred, M0inferred, True, G,tstar, deltav)            
            return tv, Nmodules, ncomm, M0inferred
    else:
        print('Number of communitites inferred by regular InfoMap: ', infomap_abs(G, deltav, 1, typeAdelta, h)[0])
        

# With this piece of code we plot Figure 11c

Nc = 20
nc = 12
M0 = [k//nc for k in range(0,Nc*nc)]
Ht = nx.watts_strogatz_graph(12,6,0)
Ht = nx.convert_node_labels_to_integers(Ht, first_label = 49)
poss = nx.spring_layout(Ht, seed =3)
tv1, Nmodules1, ncomm1, M0inferred1 = InfoAbs(G,Deltav2,0.01,0.05, 50, [0.025], 'Exp',0, M0,[5])

# With this piece of code we plot Figure 11d (color needs to be modified so the obtained six 
# communities can be told apart)

Ht = nx.watts_strogatz_graph(12,6,0)
Ht = nx.convert_node_labels_to_integers(Ht, first_label = 49)
poss = nx.spring_layout(Ht, seed =3)
tv2, Nmodules2, ncomm2, M0inferred2 = InfoAbs(G,Deltav3,0.01,0.05, 50, [0.025], 'Exp',0, M0,[5])

Ht = nx.watts_strogatz_graph(12,6,0)
Ht = nx.convert_node_labels_to_integers(Ht, first_label = 49)
poss = nx.spring_layout(Ht, seed =3)
tv0, Nmodules0, ncomm0, M0inferred0 = InfoAbs(G,Deltav1,0.01,0.05, 50, [0.025], 'Exp',0, M0,[5])


# With this piece of code, we plot Figure 11 a

plt.rcParams.update({'font.size': 15})
plt.plot(tv0, Nmodules0,'g-.',tv1,Nmodules1,'r',tv2,Nmodules2,'b--')
plt.legend(['Initial stage','Peak-duration stage','Final stage'])
plt.ylabel('Number of communities')
plt.xlabel(r'$t$')
plt.tight_layout()
plt.savefig('ncomm_all_network.eps',format = 'eps',dpi = 1000)

plt.clf()

# With this piece of code, we plot Figure 11 b

plt.rcParams.update({'font.size': 15})
plt.plot(tv0, ncomm0,'g-.',tv1,ncomm1,'r',tv2,ncomm2,'b--')
plt.legend(['Initial stage','Peak-duration stage','Final Stage'])
plt.ylabel('Number of subcommunities\nwithin a planted community')
plt.xlabel(r'$t$')
plt.tight_layout()
plt.savefig('ncomm_deltavcomm5.eps',format = 'eps',dpi = 1000)
