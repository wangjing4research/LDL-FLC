import numpy as np
from LLE import barycenter_kneighbors_graph
from scipy.optimize import minimize


#D: label distribution matrix
#P: probability matrix
def solve_Z(D, P):
    DP_0 = D * P
    DP_1 = D - DP_0
    Z_0 = barycenter_kneighbors_graph(DP_0.T).T
    Z_1 = barycenter_kneighbors_graph(DP_1.T).T
    return Z_0, Z_1




#D: label distribution matrix
#Z_0 and Z_1: manifold matrix
#rho: margin to away from 0.5

def solve_P(D, Z_0, Z_1, rho, l):   
    
    def margin(P):
        P_hat = np.abs(P - 0.5)
        sign = np.sign(0.5 - P)
        ind = P_hat < rho

        grad = sign * np.array(ind, dtype=int)
        loss = (rho - P_hat[ind]).sum()

        return loss, grad

    def obj(P):
        if l == 0:
            loss1, grad1 = 0, 0
        else:
            loss1, grad1 = margin(P)
        
        P = P.reshape((-1, 1))
        DP_0 = D * (P@one_vec.T)
        DP_1 = D - DP_0

        DPZ_0 = DP_0@IZ_0
        DPZ_1 = DP_1@IZ_1

        loss = (DPZ_0 ** 2).sum() + (DPZ_1 ** 2).sum()
        grad = (D * (DPZ_0@IZ_0.T - DPZ_1@IZ_1.T))@one_vec

        return 0.5 * loss + l * loss1, grad.reshape((-1, )) + l * grad1
    
    
    # do some initialization
    n, m = D.shape
    I = np.eye(m)
    one_vec = np.ones((m, 1))
    
    IZ_0 = I - Z_0
    IZ_1 = I - Z_1
    
    P_0 = np.random.random(n)
    bnds = tuple((0.0, 1.0) for p0 in P_0)  
    optimize_result = minimize(obj, P_0, method = 'l-BFGS-b', 
                    jac = True, bounds = bnds, 
                    #callback=fun, 
                    options = {'gtol':1e-6, 'disp': False, 'maxiter':300 })
    
    return optimize_result.x.reshape((-1, 1))


#do a bipartition
def bipart(D, inds, rho, l, iters = 100):
    #for convergence
    losses = []
    
    n = D.shape[0]
    P = np.random.random((n, 1))
    loss, _ = LDM_loss(D, P)
    losses.append(loss)
    
    for i in range(iters):
        Z_0, Z_1 = solve_Z(D, P)
        P = solve_P(D, Z_0, Z_1, rho, l)
        loss, _ = LDM_loss(D, P)
        losses.append(loss)

    return losses, P.reshape((-1, ))



def LDM_loss(D, P = None):
    
    if not P is None:
        D = D * P.reshape((-1, 1))
    Z = barycenter_kneighbors_graph(D.T).T
    loss = ((D - D@Z) ** 2).sum()
    
    return loss, Z



class LDM_SC:
    def __init__(self, Y, r = 100, rho = 0.1, l = 1):
        self.Y = Y
        self.r = r #minimal number of samples to split
        self.l = l #regularization for learning P-matrix
        self.rho = rho
                
    def partition(self, inds):
        D = self.Y[inds]
        loss, Z = LDM_loss(D) 
        
        if len(inds) < self.r:
            self.clusters.append(inds)
            self.manifolds.append(Z)
            return
    
        P = bipart(D, inds, self.rho, self.l)
        
        #return probability
        #return P
        
        #loss after partition
        loss_0, _ = LDM_loss(D, P)
        loss_1, _ = LDM_loss(D, 1 - P)
        if loss <= loss_0 + loss_1: 
            self.clusters.append(inds)
            self.manifolds.append(Z)
            return 
        
        #patition
        inds0 = inds[(P > 0.5)]
        inds1 = inds[(P <= 0.5)]
        if len(inds0) == 0 or len(inds1) == 0:
            self.clusters.append(inds)
            self.manifolds.append(Z)
            return 
        
        self.partition(inds0)
        self.partition(inds1)
        
    def solve(self):
        
        self.clusters = []
        self.manifolds = []
        
        n = self.Y.shape[0]
        inds = np.arange(n)
        self.partition(inds)
    
        cluster_labels = np.empty(n)
        for j, cluster in enumerate(self.clusters):
            cluster_labels[cluster] = j
        
        return cluster_labels, self.manifolds