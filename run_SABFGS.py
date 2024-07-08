from ldl_models import bfgs_ldl
import numpy as np
from ldl_metrics import score
import multiprocessing 
from util import *



def run_sabfgs(i, dataset, train_x, train_y, test_x, test_y, scores):
    
    model = bfgs_ldl()
    model.fit(train_x, train_y)
    y_pre = model.predict(test_x)
    val = score(test_y, y_pre)
    
    key = str(model)
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
        
        run_sabfgs(i, dataset, train_x, train_y, test_x, test_y, scores)
        
    save_dict(dataset, scores, "BFGSLDL1.pkl")


if __name__ == "__main__":
    datasets = ["SJAFFE", "M2B", "Movie", "RAF_ML", "Flickr_ldl", "Ren", "fbp5500", 
                "Gene", "SBU_3DFE", "SCUT_FBP", "Scene", "Twitter_ldl"]
    
    datasets = ["mediamill"]

    for dataset in datasets:
        run_KF(dataset)
