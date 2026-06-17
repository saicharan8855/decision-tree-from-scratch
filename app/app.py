import os
import sys
import pickle
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
COMPONENTS_DIR = os.path.join(APP_DIR, 'components')

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, COMPONENTS_DIR)

from prediction_tab import render_prediction_tab
from tree_viz_tab import render_tree_viz_tab
from metrics_tab import render_metrics_tab

st.set_page_config(page_title='Breast Cancer Decision Tree App', layout='wide')

@st.cache_resource
def load_model():
    model_path = os.path.join(PROJECT_ROOT, 'Results', 'models', 'best_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title('Breast Cancer Decision Tree App')
st.markdown(
    'This app loads a pre-trained decision tree model and lets you explore '
    'predictions, tree structure, and evaluation metrics without retraining '
    'on every interaction.'
)

tab_prediction, tab_tree, tab_metrics = st.tabs(
    ['Prediction', 'Tree Visualization', 'Metrics']
)

with tab_prediction:
    render_prediction_tab(model)

with tab_tree:
    render_tree_viz_tab(model)

with tab_metrics:
    render_metrics_tab(model)