import numpy as np
from cart_nodes import LeafNode , InternalNode

def accuracy(y_true , y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size ==0 or y_true.shape[0] == 0 or y_true.shape[0] != y_pred.shape[0]:
        return 0.0
    return float(np.mean(y_true == y_pred))

def precision(y_true , y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size ==0 or y_true.shape[0] == 0 or y_true.shape[0] != y_pred.shape[0]:
        return 0.0

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    denom = tp+ fp 
    if denom == 0:
        return 0.0

    return float(tp / denom) 

def recall(y_true , y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size == 0 or y_true.shape[0] == 0 or y_true.shape[0] != y_pred.shape[0]:
        return 0.0
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    denom = tp + fn
    if denom == 0:
        return 0.0

    return float(tp / denom)

def f1_score(y_true , y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size == 0 or y_true.shape[0] == 0 or y_true.shape[0] != y_pred.shape[0]:
        return 0.0 
    p = precision(y_true , y_pred)
    r = recall(y_true , y_pred)
    denom = p + r
    if denom == 0:
        return 0.0
    return (2* (p * r) / denom)

def mse(y_true , y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred) 

    if y_true.size == 0 or y_pred.size == 0 or y_true.shape[0] == 0 or y_true.shape[0] != y_pred.shape[0]:
        return 0.0
    
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true , y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size == 0 or y_true.shape[0] == 0 or y_true.shape[0] != y_pred.shape[0]:
        return 0.0
    
    return float(np.mean(np.abs(y_true - y_pred)))

def confusion_matrix(y_true , y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size == 0 or y_true.shape[0] == 0 or y_true.shape[0] != y_pred.shape[0]:
        return 0.0
    
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([[tn , fp] , [fn , tp]])

def print_tree(node , feature_names = None , depth = 0):
    indent = "  " * depth

    if isinstance(node , LeafNode):
        print(f"{indent}Leaf: value = {node.value}")
        return 
    if feature_names is not None:
        feature_name = feature_names[node.feature_index]
    else:
        feature_name = f"feature[{node.feature_index}]"

    print(f"{indent}If {feature_name} < {node.threshold}:")
    print_tree(node.left, feature_names = feature_names , depth = depth + 1)
    print(f"{indent}Else:")
    print_tree(node.right , feature_names = feature_names , depth = depth + 1)
