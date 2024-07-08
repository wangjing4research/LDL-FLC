from ldllrr import LDL_LRR
import numpy as np
from ldl_metrics import score
import multiprocessing 
from util import *


def do_LDLLRR(params):
    train_x, train_y, test_x, test_y, lam, beta = params
    key = "LDLLRR_" + str(lam) + "_" + str(beta)
    lrr = LDL_LRR(lam=1e-2, beta=1).fit(train_x, train_y)
    y_pre = lrr.predict(test_x)
    
    val = score(test_y, y_pre)
    return (key, val)



def run_LDLLRR(i, dataset, train_x, train_y, test_x, test_y, scores):
    
    Lam = [0.001]  #{0.00001, 0.0001, 0.001, 0.01, 0.1} 
    Beta = [1] #{0.001, 0.01, 0.1, 1, 10, 100}
    
    
    params = [(train_x, train_y, test_x, test_y, lam, beta)
              for lam in Lam for beta in Beta]
   
    print("max number of models", len(params))
    #pool = multiprocessing.Pool(1)
   
    #pool.imap_unordered
    for (key, val) in map(do_LDLLRR, params):
    
        if not key in scores.keys():
            scores[key] = []
        scores[key].append(val)

    #pool.close()
    #pool.join()

        

def run_KF(dataset):
    
    X, Y = np.load(dataset + "//feature.npy"), np.load(dataset + "//label.npy")
    train_inds = load_dict(dataset, "train_inds")
    test_inds = load_dict(dataset, "test_inds")
    
    scores = dict()
    for i in range(10):
        print(dataset, "fold", i + 1)
        
        train_x, train_y = X[train_inds[i]], Y[train_inds[i]]
        test_x, test_y = X[test_inds[i]], Y[test_inds[i]]
        
        run_LDLLRR(i, dataset, train_x, train_y, test_x, test_y, scores)
        
    save_dict(dataset, scores, "LDLLRR.pkl")


if __name__ == "__main__":
    datasets = ["SJAFFE", "M2B", "Movie", "RAF_ML", "Flickr_ldl", "Ren", "fbp5500", 
                "Gene", "SBU_3DFE", "SCUT_FBP", "Scene", "Twitter_ldl"]

    for dataset in datasets:
        run_KF(dataset)
