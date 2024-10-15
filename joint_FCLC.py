#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from util import save_dict, load_dict
#import skfuzzy as fuzz
from LLE import barycenter_kneighbors_graph
import math
import os
import multiprocessing

eps = 1e-15


# In[61]:


#D: label distribution matrix
#P: probability matrix
def solve_Z(D, P):
    if len(np.shape(P)) == 1:
        P = P.reshape(-1, 1)
    
    DP = D * P
    Z = barycenter_kneighbors_graph(DP.T).T
    return Z


# In[60]:


def fuzzy_cmeans(D, g):
    _, U, _, _, _, _, _ = fuzz.cluster.cmeans(D.T, g, 2, error=0.005, maxiter=1000)
    return U.T


# In[93]:


#D: label distribution matrix
#U: fuzzy membership matrix
#return I-Z
def solve_LDM(D, U):
    manifolds = []
    I = np.eye(D.shape[1])
    
    for j in range(U.shape[1]):
        manifolds.append(I - solve_Z(D, U[:, j]))
        
    return manifolds



# In[94]:


def update_membership(D, g, manifolds, m = 2):
    dis = ((D@manifolds) ** 2).sum(2).T
    dis = np.sqrt(dis)

    U = np.empty((D.shape[0], g))
    for j in range(g):
        U[:, j] = 1.0 / ((dis[:, j].reshape(-1, 1) / dis) ** (2 / (m-1))).sum(1)
    
    return U


# In[122]:

def joint_fc_ldm(D, g, max_iters = 150, error = 0.001):
    U_random = np.random.random((D.shape[0], g))
    U_0 = U_random / np.reshape(U_random.sum(1), (-1, 1))
    
    for i in range(max_iters):
        manifolds = solve_LDM(D, U_0)
        U_1 = update_membership(D, g, manifolds)
        
        current_error = math.sqrt(((U_1 - U_0) ** 2).sum())
        if current_error < error:
            #print(i, "break")
            break
        else:
            U_0 = U_1
        
        
    return U_1, manifolds



def get_fuzzy_manifolds(train_x, train_y, g):
    U, manifolds = joint_fc_ldm(train_y, g)
    return (U, manifolds)
