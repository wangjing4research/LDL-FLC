import numpy as np
from util import save_dict, load_dict
from ldl_metrics import score
import skfuzzy as fuzz
from LLE import barycenter_kneighbors_graph
from scipy.special import softmax
from scipy.optimize import approx_fprime
from scipy.optimize import minimize

eps = 1e-15


# In[8]:


#D: label distribution matrix
#P: probability matrix
def solve_Z(D, P):
    if len(np.shape(P)) == 1:
        P = P.reshape(-1, 1)
    
    DP = D * P
    Z = barycenter_kneighbors_graph(DP.T).T
    return Z


# In[4]:


def fuzzy_cmeans(data, g):
    _, U, _, _, _, _, _ = fuzz.cluster.cmeans(data.T, g, 2, error=0.005, maxiter=1000)
    return U.T


# In[263]:


#D: label distribution matrix
#U: fuzzy membership matrix
#return I-Z
def solve_LDM(D, U):
    manifolds = []
    I = np.eye(D.shape[1])
    
    for j in range(U.shape[1]):
        manifolds.append(I - solve_Z(D, U[:, j]))
        
    return manifolds


# In[269]:


def KL(y, y_hat):
    y = np.clip(y, eps, 1)
    y_hat = np.clip(y_hat, eps, 1)

    loss = -1 * np.sum(y * np.log(y_hat))
    grad = y_hat - y
    
    return loss, grad



class LDL_FLC:
    
    def __init__(self, g, l1, l2):
        self.g = g
        self.l1 = l1
        self.l2 = l2
        
        
    def fit(self, x, y, U = None, manifolds = None):
        self.train_x = x
        self.train_y = y
        
        if U is None:
            self.U = fuzzy_cmeans(self.train_y, self.g)
        else:
            self.U = U
            
        if manifolds is None:
            self.manifolds = solve_LDM(self.train_y, self.U)
        else:
            self.manifolds = manifolds
            
        self.n_features = self.train_x.shape[1]
        self.n_outputs = self.train_y.shape[1]
        
        self.J = np.ones((self.n_outputs, self.n_outputs))
        
    
    def fun(self, W):
        l, g = self.objective_func(W)
        print(l)
      
    def objective_func(self, W):
        W = W.reshape(self.n_features, self.n_outputs)
        y_hat = softmax(np.dot(self.train_x, W), axis = 1)
        
        loss, grad = KL(self.train_y, y_hat)
        if self.l2 !=0:
            for j in range(self.g):
                ldm_loss, ldm_grad = self.LDM(y_hat, self.U[:, j].reshape((-1, 1)), self.manifolds[j])
                
                loss += self.l2 * ldm_loss
                grad += self.l2 * ldm_grad
        
        grad = self.train_x.T@grad
        
        if self.l1 != 0:
            loss += 0.5 * self.l1 * (W **2).sum()
            grad += self.l1 * W
            
        return loss, grad.reshape((-1, ))
        
            
    
    def LDM(self, y_hat, u, I_Z):
        H = u * (u * y_hat) @ np.dot(I_Z, I_Z.T)
        
        loss = 0.5 * ((u * np.dot(y_hat, I_Z)) ** 2).sum()
        gradient = (H - (H*y_hat)@self.J) * y_hat

        return loss, gradient
    
    
            
    def solve(self, max_iters = 600):
        
        #optimize using l-bfgs
        weights = np.eye(self.n_features, self.n_outputs).reshape((-1, ))
        optimize_result = minimize(self.objective_func, weights, method = 'l-BFGS-b', 
                                   jac = True, #callback = self.fun, 
                                   options = {'gtol':1e-6, 'disp': False, 'maxiter':max_iters })
        
        weights = optimize_result.x
        self.W = weights.reshape(self.n_features, self.n_outputs)
        
    
    def predict(self, test_x):
        return softmax(test_x @ self.W, axis = 1)
    
    
    
    def __str__(self):
        model = "LDLFLC_"
        model += str(self.g) + "_"
        model += str(self.l1) + "_"
        model += str(self.l2)
        
        return model
        
        

