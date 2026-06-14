import numpy as np
from impurity import get_impurity

def find_best_threshold(feature_values , y , criterion):
    feature_values = np.asarray(feature_values)
    y = np.asarray(y)

    if len(y) < 2:
        return None , None
    
    sorted_indices = np.argsort(feature_values)
    sorted_features = feature_values[sorted_indices]
    sorted_y = y[sorted_indices] 

    parent_impurity = get_impurity(sorted_y , criterion)

    unique_vals = np.unique(sorted_features) 
    if len(unique_vals) < 2:
        return None , None 
    
    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2 

    best_threshold = None
    best_gain = -np.inf 

    for threshold in thresholds:
        left_mask = sorted_features < threshold 
        right_mask = ~left_mask 

        if not np.any(left_mask) or not np.any(right_mask):
            continue

        y_left = sorted_y[left_mask]
        y_right = sorted_y[right_mask]

        n_left = len(y_left)
        n_right = len(y_right) 
        n_total = n_left + n_right

        weighted_child_impurity = (n_left / n_total) * get_impurity(y_left, criterion) + (n_right / n_total) * get_impurity(y_right, criterion)

        gain = parent_impurity - weighted_child_impurity 

        if gain > best_gain:
            best_gain = gain 
            best_threshold = threshold 

    if best_threshold is None or best_gain <= 0:
        return None , None
    
    return best_threshold , best_gain


def find_best_split(X , y , criterion):
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples , n_features = X.shape 

    if n_samples < 2:
        return None , None , None 
    
    best_feature_index = None
    best_threshold = None
    best_gain = -np.inf


    for feature_idx in range(n_features):
        feature_values = X[: , feature_idx]

        threshold , gain = find_best_threshold(feature_values , y , criterion)

        if threshold is not None and gain > best_gain:
            best_gain =  gain 
            best_threshold = threshold 
            best_feature_index = feature_idx 

    if best_feature_index is None:
        return None , None , None
    return best_feature_index , best_threshold , best_gain