import numpy as np
from ldl_flc import LDL_FLC, fuzzy_cmeans, solve_LDM
from joint_FCLC import get_fuzzy_manifolds
from ldl_metrics import score
import multiprocessing 
from util import *


def do_FLC(param):
    train_x, train_y, test_x, test_y, l1, l2, g, U, manifolds = param
    ldl_flc = LDL_FLC(g, l1, l2)
    ldl_flc.fit(train_x, train_y, U, manifolds)
    ldl_flc.solve()
    
    val = score(test_y, ldl_flc.predict(test_x))
    return (str(ldl_flc), val)





def run_fold(fold, train_x, train_y, test_x, test_y):
    
    l1 = 0.001
    l2 = 0.01
    g = 5
    U, manifolds = get_fuzzy_manifolds(train_x, train_y, g)

    ldl_flc = LDL_FLC(g, l1, l2)
    ldl_flc.fit(train_x, train_y, U, manifolds)
    ldl_flc.solve()
    
    val = score(test_y, ldl_flc.predict(test_x))
    print(val)
    

    
    
def run_LDLFCC(dataset):
    print(dataset)
    X, Y = np.load(dataset + "//feature.npy"), np.load(dataset + "//label.npy")
    train_inds = load_dict(dataset, "train_inds")
    test_inds = load_dict(dataset, "test_inds")
    
    for i in range(10):
        print('training ' + str(i + 1) + ' fold')
        
        train_x, train_y = X[train_inds[i]], Y[train_inds[i]]
        test_x, test_y = X[test_inds[i]], Y[test_inds[i]]
        
        run_fold(i, train_x, train_y, test_x, test_y)        
        


        
if __name__ == "__main__":
    
   run_LDLFCC("SJAFFE")

    
