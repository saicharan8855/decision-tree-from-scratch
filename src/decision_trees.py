import numpy as np
from collections import Counter 
from impurity import get_impurity
from splitting import find_best_split 
from cart_nodes import InternalNode, LeafNode

class DecisionTree:
    def __init__(
            self,
            criterion,
            max_depth = None,
            min_samples_split = 2,
            min_impurity_decrease = 0.0,
            task = 'classification'
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.task = task 
        self.root = None

    def fit(self , X , y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._build_tree(X , y , depth = 0)

    def _build_tree(self, X, y, depth):
        n_samples = len(y)

        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(y)

        if n_samples < self.min_samples_split:
            return self._make_leaf(y)

        if np.unique(y).size == 1:
            return self._make_leaf(y)

        best_feature, best_threshold, best_gain = find_best_split(X, y, self.criterion)

        if best_feature is None or best_threshold is None or best_gain is None:
            return self._make_leaf(y)

        if best_gain < self.min_impurity_decrease:
            return self._make_leaf(y)

        left_mask = X[:, best_feature] < best_threshold
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            return self._make_leaf(y)

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)

        return InternalNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            gain=best_gain
        )
    
    def _make_leaf(self, y):
        if self.task == "classification":
            counts = Counter(y)
            majority_class = counts.most_common(1)[0][0]
            return LeafNode(value=majority_class, class_counts=dict(counts))

        if self.task == "regression":
            mean_value = float(np.mean(y))
            return LeafNode(value=mean_value, class_counts=None)

        raise ValueError("task must be 'classification' or 'regression'")

    def predict(self, X):
        X = np.asarray(X)
        predictions = [self._predict_single(row, self.root) for row in X]
        return np.asarray(predictions)

    def _predict_single(self, x, node):
        if isinstance(node, LeafNode):
            return node.value

        if x[node.feature_index] < node.threshold:
            return self._predict_single(x, node.left)

        return self._predict_single(x, node.right)



