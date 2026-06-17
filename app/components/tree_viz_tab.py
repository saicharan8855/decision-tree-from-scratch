import streamlit as st
import io
import contextlib
from utils import print_tree

FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

def render_tree_viz_tab(model):
    st.header("Tree Structure")
    st.markdown("Text representation of the trained decision tree. Each branch shows the feature and threshold used to split.")

    max_depth_view = st.slider("Max depth to display", min_value=1, max_value=10, value=5)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _print_tree_limited(model.root, FEATURE_NAMES, depth=0, max_depth=max_depth_view)

    st.code(buffer.getvalue(), language="text")


def _print_tree_limited(node, feature_names, depth, max_depth):
    from cart_nodes import LeafNode
    indent = "  " * depth

    if isinstance(node, LeafNode):
        print(f"{indent}Leaf: value = {node.value}")
        return

    if depth >= max_depth:
        print(f"{indent}... (truncated)")
        return

    feature_name = feature_names[node.feature_index]
    print(f"{indent}If {feature_name} < {node.threshold:.4f}:")
    _print_tree_limited(node.left, feature_names, depth + 1, max_depth)
    print(f"{indent}Else:")
    _print_tree_limited(node.right, feature_names, depth + 1, max_depth)