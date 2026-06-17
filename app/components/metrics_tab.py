import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import accuracy, precision, recall, f1_score, confusion_matrix

def render_metrics_tab(model):
    st.header("Model Performance")

    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
    data_dir = os.path.join(PROJECT_ROOT, 'Data', 'processed')

    X_test = np.load(os.path.join(data_dir, 'bc_X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'bc_y_test.npy'))

    y_pred = model.predict(X_test)

    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{prec:.4f}")
    col3.metric("Recall", f"{rec:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Pred 0', 'Pred 1'],
        yticklabels=['Actual 0', 'Actual 1'],
        ax=ax
    )
    st.pyplot(fig)

    st.markdown(
        f"Out of {len(y_test)} test samples, the model correctly classified "
        f"{int(np.sum(y_test == y_pred))} ({acc*100:.2f}%)."
    )