from typing import List , Dict , Optional , Union , Any

class InternalNode:
    def __init__(
        self,
        feature_index : int,
        threshold : float,
        left : Optional[Union["InternalNode", "LeafNode"]],
        right : Optional[Union["InternalNode", "LeafNode"]],
        impurity : float,
        n_samples : int,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.impurity = impurity
        self.n_samples = n_samples

    def __repr__(self) -> str:
        return (
            f"InternalNode("
            f"feature_index={self.feature_index}, "
            f"threshold={self.threshold}, "
            f"impurity={self.impurity}, "
            f"n_samples={self.n_samples}"
            f")"
        )


class LeafNode:
    def __init__(
        self,
        value: Any,
        n_samples: int,
        class_counts: Optional[Dict[Any, int]] = None,
    ) -> None:
        self.value = value
        self.n_samples = n_samples
        self.class_counts = class_counts

    def __repr__(self) -> str:
        return (
            f"LeafNode("
            f"value={self.value}, "
            f"n_samples={self.n_samples}, "
            f"class_counts={self.class_counts}"
            f")"
        )