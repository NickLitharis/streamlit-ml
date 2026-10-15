"""Tests for the ML logic in ml.py.

These drive ``perform_ml()`` directly, so they run headless — no Streamlit server
and no browser are needed.

Run with::

    python -m pytest tests -v
"""
import numpy as np
import pandas as pd
import pytest

from ml import perform_ml

EXPECTED_COLUMNS = {
    "Method",
    "Accuracy/silhouette score",
    "Precision",
    "Recall",
    "F1 score",
    "Calinski-Harabasz index",
}

SUPERVISED = 0
UNSUPERVISED = 1


def blobs(n_per_class=40, classes=2, features=4, spread=0.4, seed=0):
    """Well-separated Gaussian blobs; the last column is the label."""
    rng = np.random.default_rng(seed)
    blocks = []
    for label in range(classes):
        points = rng.normal(loc=label * 6.0, scale=spread, size=(n_per_class, features))
        blocks.append(np.column_stack([points, np.full(n_per_class, label)]))
    data = np.vstack(blocks)
    rng.shuffle(data)
    columns = [f"f{i}" for i in range(features)] + ["label"]
    return pd.DataFrame(data, columns=columns)


def three_clusters_two_classes(n_per_cluster=40, seed=2):
    """Three separated groups in feature space, but only two class labels.

    Binary labels keep it inside perform_ml's supported path while still giving
    the clustering something to get right (or wrong).
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for centre, label in zip((0.0, 6.0, 12.0), (0, 0, 1)):
        points = rng.normal(loc=centre, scale=0.3, size=(n_per_cluster, 4))
        blocks.append(np.column_stack([points, np.full(n_per_cluster, label)]))
    data = np.vstack(blocks)
    rng.shuffle(data)
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(4)] + ["label"])


def xor_frame(n=200, seed=1):
    """Labels no single split can separate, so tree depth must matter."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 2))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)
    return pd.DataFrame(np.column_stack([x, y]), columns=["x1", "x2", "label"])


def supervised(frame, **kwargs):
    return perform_ml(frame, **kwargs).iloc[SUPERVISED]


def unsupervised(frame, **kwargs):
    return perform_ml(frame, **kwargs).iloc[UNSUPERVISED]


def test_returns_both_rows_with_the_documented_columns():
    result = perform_ml(blobs())
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == EXPECTED_COLUMNS
    assert list(result["Method"]) == ["Supervised learning", "Unsupervised learning"]


def test_supervised_metrics_are_probabilities():
    row = supervised(blobs())
    for column in ("Accuracy/silhouette score", "Precision", "Recall", "F1 score"):
        value = row[column]
        assert 0.0 <= value <= 1.0, f"{column} out of range: {value}"


def test_unsupervised_row_reports_cluster_quality_only():
    row = unsupervised(blobs())
    assert -1.0 <= row["Accuracy/silhouette score"] <= 1.0  # silhouette
    assert row["Calinski-Harabasz index"] > 0
    for column in ("Precision", "Recall", "F1 score"):
        assert pd.isna(row[column]), f"{column} should be NaN on the unsupervised row"


def test_supervised_row_has_no_cluster_metric():
    assert pd.isna(supervised(blobs())["Calinski-Harabasz index"])


def test_separable_data_is_classified_accurately():
    row = supervised(blobs(spread=0.3))
    assert row["Accuracy/silhouette score"] > 0.9
    assert row["F1 score"] > 0.9


def test_max_depth_reaches_the_classifier():
    """On XOR data a stump cannot separate the classes; a deep tree can.

    The model has no random_state, so run-to-run accuracies wobble by a percent
    or two — assert against absolute thresholds rather than only comparing the
    two runs, or noise alone can satisfy the comparison.
    """
    frame = xor_frame()
    shallow = supervised(frame, max_d=1)["Accuracy/silhouette score"]
    deep = supervised(frame, max_d=8)["Accuracy/silhouette score"]
    assert shallow < 0.8, f"a depth-1 stump should not fit XOR data (got {shallow})"
    assert deep > 0.9, f"a depth-8 tree should fit XOR data (got {deep})"
    assert deep > shallow


def test_n_clusters_reaches_the_clustering():
    """With three separated groups, three clusters should fit decidedly better than two."""
    frame = three_clusters_two_classes()
    two = unsupervised(frame, n_clu=2)["Accuracy/silhouette score"]
    three = unsupervised(frame, n_clu=3)["Accuracy/silhouette score"]
    assert three > two + 0.2, f"n_clu had no real effect (k=2: {two}, k=3: {three})"


@pytest.mark.xfail(
    reason="precision_score/recall_score/f1_score default to average='binary', "
    "which raises on more than two classes — the app cannot handle multiclass input",
    strict=False,
)
def test_multiclass_labels_are_supported():
    perform_ml(blobs(classes=3))
