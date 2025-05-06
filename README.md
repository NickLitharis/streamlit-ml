# ML Data Explorer: Classification & Clustering Web App

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

An interactive web application for performing supervised classification and unsupervised clustering on tabular data, built with Python and Streamlit.

## Features

- **Dual ML Functionality**:
  - ✅ Supervised Learning: Random Forest Classifier
  - ✅ Unsupervised Learning: Hierarchical Clustering
- **Interactive UI**:
  - File upload for CSV/TXT datasets
  - Adjustable parameters via sliders
- **Comprehensive Evaluation**:
  - Classification metrics: Accuracy, Precision, Recall, F1-score
  - Clustering metrics: Silhouette Score, Calinski-Harabasz Index

## Installation

1. Clone the repository:
   ```bash
   git clone [your-repo-url]
   cd [repo-name]
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. In the web interface:

    - Upload your dataset (CSV or TXT format)

    - Adjust parameters using the sidebar sliders:

        - Max Depth (for Random Forest)

        - Number of Clusters (for Hierarchical Clustering)

    - View evaluation metrics in the results table

## Code Structure
```plaintext
    app.py
    ├── perform_ml(df, max_d, n_clu)
    │   ├── Supervised Learning (RandomForestClassifier)
    │   └── Unsupervised Learning (AgglomerativeClustering)
    └── app()
        ├── File upload handler
        ├── Parameter sliders
        └── Results display
```

## Example Dataset

For best results, provide datasets with:
- Supervised Learning: Labeled data (features + target variable in last column)
- Unsupervised Learning: Feature columns only

## Future Enhancements

- Add more algorithms (K-Means, SVM, etc.)

- Visualizations (Confusion matrix, Dendrogram)

- Data preprocessing options

- Export results functionality
