# ML Data Explorer

Upload any CSV or TXT dataset and instantly run both supervised and unsupervised machine learning — no code required.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit_learn-F7931E?logo=scikit-learn&logoColor=white)

---

## What It Does

Upload a dataset and the app runs two models simultaneously:

| Method | Algorithm | Metrics |
|---|---|---|
| Supervised | Random Forest Classifier | Accuracy, Precision, Recall, F1 |
| Unsupervised | Agglomerative Clustering | Silhouette Score, Calinski-Harabasz Index |

Results are displayed side-by-side in a table. Use the sidebar sliders to tune parameters without touching code.

---

## Parameters

| Parameter | Default | Range |
|---|---|---|
| Max depth (Random Forest) | 2 | 1–10 |
| Number of clusters | 2 | 2–10 |

---

## Installation

```bash
git clone https://github.com/NickLitharis/streamlit-ml
cd streamlit-ml
pip install -r requirements.txt
streamlit run ml.py
```

---

## Data Format

- **Supervised learning** — CSV/TXT with features in all columns except the last; last column = target label
- **Unsupervised learning** — feature columns only, no target needed
