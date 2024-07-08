import numpy as np
import os
import pickle


def save_dict(dataset, scores, name):
    if not name.endswith(".pkl"):
        name += ".pkl"

    with open(dataset + "//" + name, 'wb') as f:
        pickle.dump(scores, f)

def load_dict(dataset, name):
    if not os.path.exists(dataset + "//" + name):
        name += ".pkl"
        
    with open(dataset + "//" + name, 'rb') as f:
        return pickle.load(f)
    
    
    

    
