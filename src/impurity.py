import numpy as np

def gini(y):

    y = np.asarray(y)

    if y.size == 0:
        return 0 
    
    if y.ndim != 1:
        y = np.ravel(y) 

    _ , counts = np.unique(y , return_counts = True)
    probs = counts.astype(np.float64) / counts.sum() 

    impurity = 1.0 - np.sum(probs ** 2) 

    return float(impurity)  

def entropy(y):
    y = np.asarray(y)

    if y.size == 0:
        return 0
    
    if y.ndim != 1:
        y = np.ravel(y) 

    _ , counts = np.unique(y , return_counts = True)

    probs = counts.astype(np.float64) / counts.sum() 
    probs = probs[probs > 0] 

    entropy_bits = -np.sum(probs * np.log2(probs))

    return float(entropy_bits) 

def variance_reduction(y):

    y = np.asarray(y) 

    if y.size == 0:
        return 0
    
    if y.ndim != 1:
        y = np.ravel(y) 

    mean_y = np.mean(y) 
    variance = np.sum((y - mean_y) ** 2 )/len(y)

    return float(variance)