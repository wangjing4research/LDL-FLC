import numpy as np
from ldl_metrics import score
import multiprocessing 
from SCL import LDL_SCL
import sys
from util import *

def do_LDL_SCL(params):
    train_x, train_y, test_x, test_y, l1, l2, l3, c = params
    key = "LDLSCL_" + str(l1) + "_" + str(l2) + "_" + str(l3) + "_" + str(c)
    y_pre = LDL_SCL(train_x, train_y, test_x, test_y, l1, l2, l3, c)
    val = score(test_y, y_pre)
    
    return (key, val)


def tune_LDL_SCL(i, dataset, train_x, train_y, test_x, test_y, scores):
    L1 = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    groups = range(14)
   
    print(groups)
    params = [(train_x, train_y, test_x, test_y, l1, l2, l3, g)
              for l1 in L1 for l2 in L1 for l3 in L1 for g in groups]
   
    print("max number of models", len(params))
    pool = multiprocessing.Pool(8)
   
    finished = 0
   
    for (key, val) in pool.imap_unordered(do_LDL_SCL, params):
        finished += 1
        if finished % 500 == 0:
            print(dataset, i, finished, len(params))
    
        if not key in scores.keys():
            scores[key] = []
        scores[key].append(val)

    pool.close()
    pool.join()


    
def run_LDLSCL(i, dataset, train_x, train_y, test_x, test_y, scores):
    
    if dataset in ["Gene", "Movie"]:
        l1 = 0.00001
        l2 = 0.001
        l3 = 0.0001
        c = 12
    elif dataset in ["M2B","SCUT_FBP", "SBU_3DFE","Scene", "SJAFFE"]:
        l1 = 0.0001
        l2 = 0.001
        l3 = 0.001
        if dataset == "SJAFFE":
            c = 5
        else:
            c = 8
    elif dataset in ["fbp5500", "RAF_ML", "Twitter_ldl", "Flickr_ldl", "Ren"]:
        l1 = 0.001
        l2 = 0.001
        l3 = 0.001
        c = 5
    
    y_pre = LDL_SCL(train_x, train_y, test_x, test_y, l1, l2, l3, c)
    key = "LDLSCL_" + str(l1) + "_" + str(l2) + "_" + str(l3) + "_" + str(c)
    val = score(test_y, y_pre)
    if not key in scores.keys():
        scores[key] = []
    scores[key].append(val)
        

def run_KF(dataset):
    
    X, Y = np.load(dataset + "//feature.npy"), np.load(dataset + "//label.npy")
    train_inds = load_dict(dataset, "train_inds")
    test_inds = load_dict(dataset, "test_inds")
    
    scores = dict()
    for i in range(10):
        print(dataset, "fold", i + 1)
        
        train_x, train_y = X[train_inds[i]], Y[train_inds[i]]
        test_x, test_y = X[test_inds[i]], Y[test_inds[i]]
        
        #run_LDLSCL(i, dataset, train_x, train_y, test_x, test_y, scores)
        tune_LDL_SCL(i, dataset, train_x, train_y, test_x, test_y, scores)
        
    save_dict(dataset, scores, "LDLSCL.pkl")


if __name__ == "__main__":
    datasets = ["SJAFFE", "M2B", "Movie", "RAF_ML", "Flickr_ldl", "Ren", "fbp5500", 
                "Gene", "SBU_3DFE", "SCUT_FBP", "Scene", "Twitter_ldl"]

    datasets = ["SJAFFE"]
    for dataset in datasets:
        if len(sys.argv) == 1:
            run_KF(dataset)
        else:
            run_KF(dataset, float(sys.argv[1]))
