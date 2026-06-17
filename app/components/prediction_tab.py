import streamlit as st
import numpy as np

FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

DEFAULT_VALUES = [
    14.1, 19.3, 92.0, 655.0, 0.096,
    0.104, 0.089, 0.049, 0.181, 0.063,
    0.405, 1.22, 2.87, 40.3, 0.007,
    0.025, 0.032, 0.012, 0.020, 0.004,
    16.3, 25.7, 107.3, 880.6, 0.132,
    0.254, 0.272, 0.115, 0.290, 0.084
]

def render_prediction_tab(model):
    st.header("Predict Diagnosis")
    st.markdown("Adjust feature values below and click **Predict** to classify the sample.")

    values = []
    cols = st.columns(3)
    for i, name in enumerate(FEATURE_NAMES):
        col = cols[i % 3]
        with col:
            val = st.number_input(
                label=name,
                value=float(DEFAULT_VALUES[i]),
                format="%.4f",
                key=f"feat_{i}"
            )
            values.append(val)

    if st.button("Predict"):
        X = np.array(values).reshape(1, -1)
        prediction = model.predict(X)[0]

        if prediction == 0:
            st.success("Prediction: Benign (0)")
        else:
            st.error("Prediction: Malignant (1)")

        leaf = _get_leaf(model.root, X[0])
        if leaf is not None and getattr(leaf, "class_counts", None):
            st.write("Leaf class distribution:", leaf.class_counts)


def _get_leaf(node, x):
    from cart_nodes import LeafNode
    while not isinstance(node, LeafNode):
        if x[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node