import numpy as np
import copy
from collections import Counter 
from decision_trees import LeafNode , InternalNode


def _count_leaves(node):
    if isinstance(node , LeafNode):
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right) 

def _majority_class(y):
    counts = Counter(y)
    return counts.most_common(1)[0][0]

def _mean_value(y):
    return float(np.mean(y))

def compute_node_cost(node , X , y, task="classification"):
    y = np.asarray(y) 

    if len(y) == 0:
        return 0.0
    
    if task == "classification":
        majority = _majority_class(y)
        misclassified = np.sum(y != majority)
        return float(misclassified / len(y))

    elif task == "regression":
        mean_y = np.mean(y)
        return float(np.mean((y - mean_y) ** 2))

    else:
        raise ValueError("task must be 'classification' or 'regression'")

def compute_subtree_cost(node, X, y, task):
    X = np.asarray(X)
    y = np.asarray(y)

    if len(y) == 0:
        return 0.0

    if isinstance(node, LeafNode):
        return compute_node_cost(node, X, y, task)

    left_mask = X[:, node.feature_index] < node.threshold
    right_mask = ~left_mask

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    cost_left = compute_subtree_cost(node.left, X_left, y_left, task)
    cost_right = compute_subtree_cost(node.right, X_right, y_right, task)

    n = len(y)
    weighted_left = (len(y_left) / n) * cost_left if len(y_left) > 0 else 0.0
    weighted_right = (len(y_right) / n) * cost_right if len(y_right) > 0 else 0.0

    return weighted_left + weighted_right


def compute_effective_alpha(node, X, y, task):
    if isinstance(node, LeafNode):
        return np.inf

    num_leaves = _count_leaves(node)
    denominator = num_leaves - 1

    if denominator == 0:
        return np.inf

    R_t = compute_node_cost(node, X, y, task)
    R_T_t = compute_subtree_cost(node, X, y, task)

    return (R_t - R_T_t) / denominator

def _prune_tree(root , alpha , X , y , task = "classification"):
    root = copy.deepcopy(root)

    def _prune(node , X_node , y_node):
        if isinstance(node , LeafNode):
            return node
    
        left_mask = X_node[:, node.feature_index] < node.threshold
        right_mask = ~left_mask

        X_left , y_left = X_node[left_mask] , y_node[left_mask]
        X_right , y_right = X_node[right_mask] , y_node[right_mask]

        node.left = _prune(node.left , X_left , y_left)
        node.right = _prune(node.right , X_right , y_right)

        eff_alpha = compute_effective_alpha(node , X_node , y_node , task)

        if eff_alpha <= alpha:
            if task == "classification":
                value = _majority_class(y_node)
                counts = dict(Counter(y_node))
                return LeafNode(value=value, n_samples=len(y_node), class_counts=counts)
            else:
                value = _mean_value(y_node)
                return LeafNode(value=value, n_samples=len(y_node), class_counts=None)
        return node
    
    return _prune(root , X , y)

def cost_complexity_pruning(root , X , y , task = "classification"):

    X = np.asarray(X)
    y = np.asarray(y)

    current_tree = copy.deepcopy(root)

    path = []

    while True:
        candidates = []

        def _collect(node , X_node , y_node):
            if isinstance(node , LeafNode):
                return
            left_mask = X_node[:, node.feature_index] < node.threshold
            right_mask = ~left_mask

            X_left, y_left = X_node[left_mask], y_node[left_mask]
            X_right, y_right = X_node[right_mask], y_node[right_mask]

            eff_alpha = compute_effective_alpha(node, X_node, y_node , task)
            candidates.append(eff_alpha)

            _collect(node.left, X_left, y_left)
            _collect(node.right, X_right, y_right)

        _collect(current_tree, X, y)

        if not candidates:
            break

        min_alpha = min(candidates)
        pruned_tree = _prune_tree(current_tree , min_alpha , X , y , task = task)

        path.append((min_alpha , copy.deepcopy(pruned_tree)))
        current_tree = pruned_tree 

        if isinstance(current_tree , LeafNode):
            break 

    return path


