import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, silhouette_score, f1_score, calinski_harabasz_score


def perform_ml(df, max_d=2, n_clu=2):
    # Perform supervised learning using the RandomForest Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=max_d)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # Perform unsupervised learning using HierarchicalClustering
    X = df.iloc[:, :-1]
    agg = AgglomerativeClustering(n_clusters=n_clu)
    agg.fit(X)
    y_pred = agg.labels_
    silhouette = silhouette_score(X, y_pred)
    calinski_harabasz = calinski_harabasz_score(X, y_pred)

    # Return evaluation results
    eval_results = pd.DataFrame({
        "Method": ["Supervised learning", "Unsupervised learning"],
        "Accuracy/silhouette score": [acc, silhouette],
        "Precision": [precision, np.nan],
        "Recall": [recall, np.nan],
        "F1 score": [f1, np.nan],
        "Calinski-Harabasz index": [np.nan, calinski_harabasz]
    })
    return eval_results


def app():
    # Create file uploader widget that allows user to upload a CSV or TXT file
    uploaded_file = st.file_uploader(
        "Upload your input CSV or TXT file", type=["csv", "txt"])

    #  Create sliders for the user to change parameters
    md = st.sidebar.slider("Max depth (RandomForestClassifier)", 1, 10, 2)
    nc = st.sidebar.slider(
        "Number of clusters (HierarchicalClustering)", 2, 10, 2)

    # If file is uploaded, read the file contents and perform machine learning processes
    if uploaded_file is not None:
        file_contents = uploaded_file.read()
        # Determine the file type (CSV or TXT) based on the file extension
        file_ext = uploaded_file.name.split(".")[-1].lower()
        # Convert the file contents to a Pandas dataframe
        if file_ext == "csv":
            df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
        elif file_ext == "txt":
            df = pd.read_csv(io.StringIO(file_contents.decode(
                'utf-8')), delim_whitespace=True, quotechar='"')
        else:
            st.write("Error: Invalid file type. Please upload a CSV or TXT file.")
            return
        # Perform machine learning processes on the dataframe using the updated parameters
        eval_results = perform_ml(df, md, nc)
        # Display the evaluation results in a table
        st.subheader("Evaluation results")
        st.dataframe(eval_results)


# Run Streamlit app
if __name__ == "__main__":
    app()
