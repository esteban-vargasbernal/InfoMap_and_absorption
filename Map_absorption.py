import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from numpy.linalg import inv

### Computing the standard map. Generate Figure 5

N=3

Partitions = [ [[2],[1,3]], [[1,2,3]], [[1],[2,3]], [[3],[1,2]], [[1],[2],[3]]  ]
m = len(Partitions) # Number of partitions
Delta2 = np.arange(0.1,1,0.01)
LM = np.zeros((m,len(Delta2))) 
jj = 0
for delta2 in Delta2:
    # Define movement ajacency matrix 
    A = np.array([[0,1,1],[0,0,0],[1,1,0]])
    # Define outflow
    outfl = A.sum(axis=0)
    W = np.diag(outfl)

    # defining removal rates and teleportation
    delta1 = 0.1
    delta3 = 0.1
    tau = 0.15
    r = 0.5

    delta  = np.array([delta1,delta2,delta3])
    Lambda = np.max(outfl/delta)
    #t = r/Lambda
    t = 1/20
    Ddelta = np.diag(delta)


    # defining adjacency accounting for removal
    Adelta = np.identity(3) + t*(A-W).dot(inv(Ddelta))


    # Including teleportation:
    # In this case the jump process matrix coincides with A_\delta 
    # because A_\delta is already stochastic. However, we tell them apart
    P0 =  Adelta
    # defining preference vector, teleportation to links (in this case coincides 
    # with teleportation to nodes)
    v = 1/(np.sum(Adelta))*np.sum(Adelta,axis=0)
    P = (1-tau)*P0+tau*v.reshape(3,1).dot(np.ones(3).reshape(1,3))
    eigenValues, eigenVectors = linalg.eig(P)
    idx = eigenValues.argsort()[-1]
    pt = np.absolute(eigenVectors[:,idx])
    pstar = 1/np.sum(pt)*pt

    # Unrecording teleportation

    Qt = P0.dot(np.diag(pstar))
    Q = Qt/np.sum(Qt)

    p = np.sum(Q,axis=1)
    # Defining partition
    ii = 0
    for Mt in Partitions: 
        n=len(Mt)
        M = [0]*n
        for i in range(0,n):
            M[i] = np.array(Mt[i])-1

        # Defining entropy function 
        def H(p):
            pt = p[p !=0]
            p = pt/np.sum(pt)
            return -np.dot(p,np.log2(p))

        # Defining q_i^in, q_i^out


        if n > 1:
            qinv = []
            qoutv = []


            for i in range(0,n):
                Mi = np.asarray(M[i])
                qinv.append(np.sum(Q[Mi,np.delete(range(0,N),Mi)]))
                qoutv.append(np.sum(Q[np.delete(range(0,N),Mi),Mi]))

            qinv = np.asarray(qinv)
            qoutv = np.asarray(qoutv)


            qin = np.sum(qinv)
            qout = np.sum(qoutv)


            # Defining H(Q)
            HQ = H(qinv)       
        else:
             HQ=0; qoutv = np.array([0])   

        # Defining HP_i
        HPv = []
        for i in range(0,n):
            pt = np.concatenate((p[M[i]],np.asarray([qoutv[i]])))
            HPv.append(H(pt)*np.sum(pt))

        HP = np.sum(HPv)

        # Map equation
        LM[ii,jj] = qin*HQ+HP
        ii = ii+1
    jj=jj+1

Lstyles = ['-', '--', '-.', ':','+']
plt.rcParams.update({'font.size': 10})
plt.figure()
for i in range(0,m):
    plt.plot(Delta2,LM[i],Lstyles[i])
plt.xlabel(r'$\delta_2$', fontsize = 15)
plt.ylabel(r'$L(M)$', fontsize = 15)
plt.legend(Partitions,loc=1)
plt.savefig('ex1.eps', format='eps', dpi=1000)
plt.show()


### Computing the map for absorption. Generate Figure 3

Partitions = [ [[2],[1,3]], [[1,2,3]], [[1],[2,3]], [[3],[1,2]], [[1],[2],[3]]  ]
m = len(Partitions) # Number of partitions
Delta2 = np.arange(0.1,10.1,0.1)
LM = np.zeros((m,len(Delta2))) 
jj = 0
for delta2 in Delta2:
    
    # defining removal rates and teleportation
    delta1 = 0.1
    delta3 = 0.1
    delta  = np.array([delta1,delta2,delta3])
    Ddelta = np.diag(delta)

    
    # Define movement ajacency matrix 
    A = np.array([[0,1,1,0],[0,0,0,0],[1,1,0,0],[delta1, delta2, delta3,1]])
    A0 = np.array([[0,1/2,1],[0,0,0],[1,1/2,0]])
    # Define outflow
    outfl = A.sum(axis=0)
    W = np.diag(outfl)
    W0 = W[np.ix_([0,1,2],[0,1,2])]
    I = np.identity(3)
    
    P = A.dot(inv(W))
    
    Q = P[np.ix_([0,1,2],[0,1,2])]
    
    Nf = inv(I-Q)     
    outflowN = Nf.sum(axis=0) 
    WN = np.diag(outflowN)    
    Q = Nf.dot(inv(WN))

    # distribution pi_delta
    p = np.sum(Q,axis=1)/np.sum(np.sum(Q))
    
    # Transitions P_delta
    hw = 1
    P0 = I - hw*(W0.dot(inv(hw*W0+Ddelta)) - A0.dot(inv(hw*W0 + Ddelta)) ) 
    
    # Defining partition
    ii = 0
    for Mt in Partitions: 
        n=len(Mt)
        M = [0]*n
        for i in range(0,n):
            M[i] = np.array(Mt[i])-1

        # Defining entropy function 
        def H(p):
            pt = p[p !=0]
            p = pt/np.sum(pt)
            return -np.dot(p,np.log2(p))

        # Defining q_i^in, q_i^out


        if n > 1:
            qinv = []
            qoutv = []


            for i in range(0,n):
                Mi = np.array(M[i])
                qinv.append(np.sum(P0[Mi,np.delete(range(0,N),Mi)]))
                qoutv.append(np.sum(P0[np.delete(range(0,N),Mi),Mi]))

            qinv = np.asarray(qinv)
            qoutv = np.asarray(qoutv)


            qin = np.sum(qinv)
            qout = np.sum(qoutv)


            # Defining H(Q)
            HQ = H(qinv)       
        else:
             HQ=0; qoutv = np.array([0])   

        # Defining HP_i
        HPv = []
        for i in range(0,n):
            pt = np.concatenate((p[M[i]],np.asarray([qoutv[i]])))
            HPv.append(H(pt)*np.sum(pt))

        HP = np.sum(HPv)

        # Map equation
        LM[ii,jj] = qin*HQ+HP
        ii = ii+1
    jj=jj+1

Lstyles = ['-', '--', '-.', ':','+']
plt.rcParams.update({'font.size': 10})
plt.figure()
for i in range(0,m):
    plt.plot(Delta2,LM[i],Lstyles[i])
plt.xlabel(r'$\delta_2$', fontsize = 15)
plt.legend(Partitions,loc=1)
plt.savefig('ex1.eps', format='eps', dpi=1000)
plt.show()

