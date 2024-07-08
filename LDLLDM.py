import os
import pickle
import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix 
from ldl_metrics import score
from scipy.special import softmax
from scipy.linalg import solve
from sklearn.cluster import KMeans
from LLE import barycenter_kneighbors_graph


eps = 1e-15

def weighted_ME(Y_hat, W):  
    weighted_Y_hat = Y_hat * W
    grad = np.sum(-weighted_Y_hat, 1).reshape((-1, 1)) * Y_hat + weighted_Y_hat
    return grad

def weighted_log_ME(Y_hat, W):
        return W - W.sum(1).reshape((-1, 1)) * Y_hat
    

def KL(y, y_hat, J = None):
    y = np.clip(y, eps, 1)
    y_hat = np.clip(y_hat, eps, 1)
    
    if J is None:
        loss = -1 * np.sum(y * np.log(y_hat))
        grad = y_hat - y
    else:
        loss = -1 * np.sum(J * y * np.log(y_hat))
        grad = J * (y_hat - y)
    
    return loss, grad


def append_intercept(X):
        return np.hstack((X, np.ones(np.size(X, 0)).reshape(-1, 1)))
    

#label distribution matrix is not missing
class LDLLDM_Full: 

    #each cluster
    class Cluster:
        def __init__(self, X, l, Y, Z = None):
            
            self.X = X
            self.l = l
            self.I = np.eye(Y.shape[1])
            self.LDM(Y, Z)
          
        #learn label distribution manifold
        def LDM(self, Y, Z):
            if Z is None:
                self.Z = barycenter_kneighbors_graph(Y.T).T
            else:
                self.Z = Z
            
            self.I_Z = self.I - self.Z 
            self.L = np.dot(self.I_Z, self.I_Z.T)
        
        
        def LDL(self, Y_hat):
            if self.l == 0:
                return 0, 0
            
            loss = (np.dot(Y_hat, self.I_Z) ** 2).sum()
            grad = np.dot(self.X.T, weighted_ME(Y_hat, 2 * np.dot(Y_hat, self.L)))
            
            return self.l * loss, self.l * grad
            
    
    def __init__(self, X, Y, l1, l2, l3, g = 0, clu_labels = None, manifolds = None):
        
        self.X = append_intercept(X)        
        self.Y = Y
        
        self.l1 = l1
        self.l2 = l2 #global 
        self.l3 = l3 #local
        
        self.g = g
        
        self.n_examples, self.n_features, = self.X.shape
        self.n_outputs = self.Y.shape[1]
        
        #conduct K-means
        if clu_labels is None:
            kmeans = KMeans(n_clusters=g).fit(Y)
            clu_labels = kmeans.predict(Y)
            
        self.__init_clusters(clu_labels, manifolds)

        
    def __init_clusters(self, clu_labels, manifolds):
        self.clusters = []
        self.inds = []
        
        #global label distribution manifold 
        clu = self.Cluster(self.X, self.l2, self.Y)
        self.clusters.append(clu)
        self.inds.append(np.asarray(np.ones(self.n_examples), dtype = bool))
    
        if self.g > 1: 
            for i in range(self.g):
                ind = (clu_labels == i)
                X_i = self.X[ind]
                Y_i = self.Y[ind]
                
                if manifolds is None:
                    clu = self.Cluster(X_i, self.l3, Y_i)
                else:
                    clu = self.Cluster(X_i, self.l3, Y_i, manifolds[i])
                
                self.clusters.append(clu)
                self.inds.append(ind)
    
    def LDL(self, W):
        W = W.reshape(self.n_features, self.n_outputs)
        Y_hat = softmax(np.dot(self.X, W), axis = 1)
        
        loss, grad = KL(self.Y, Y_hat)
        grad = np.dot(self.X.T, grad)
        
        if self.l1 != 0:
            loss += 0.5 * self.l1 * (W **2).sum()
            grad += self.l1 * W
        
        for (ind, clu) in zip(self.inds, self.clusters):
            clu_l, clu_g = clu.LDL(Y_hat[ind])
            
            loss += clu_l
            grad += clu_g
        
        return loss, grad.reshape((-1, ))
        #return loss, grad
    

    
    def LDL_loss(self, W):
        l, _ = self.LDL(W)
        return l
    
    def LDL_grad(self, W):
        _, g = self.LDL(W)
        return g
    
    #only for echo
    '''
    def fun(self, W):
        l, _ = self.LDL(W)
        print(l)
    '''
    
    #optimize using pymanopt
    def solve_gd(self, max_iters = 600):
        manifold = Euclidean(self.n_features, self.n_outputs)
        problem = Problem(manifold=manifold, cost=self.LDL_loss, grad = self.LDL_grad)
        solver = SteepestDescent(max_iters)
        
        Xopt = solver.solve(problem)
        self.W = Xopt
    
    
    #optimize using l-bfgs
    def solve(self, max_iters = 600):
        
        #optimize using pymanopt        
        #self.solve_gd()

        #optimize using l-bfgs
        weights = np.eye(self.n_features, self.n_outputs).reshape((-1, ))
        optimize_result = minimize(self.LDL, weights, method = 'l-BFGS-b', jac = True, #callback = self.fun, 
                                   options = {'gtol':1e-6, 'disp': False, 'maxiter':max_iters })
        
        weights = optimize_result.x
        self.W = weights.reshape(self.n_features, self.n_outputs)

    
    def predict(self, X_test):
        return softmax(np.dot(append_intercept(X_test), self.W), axis = 1)
    
    
    def __str__(self):
        model = "LDLLDM_"
        model += str(self.l1) + "_"
        model += str(self.l2) + "_"
        model += str(self.l3)
        
        return model