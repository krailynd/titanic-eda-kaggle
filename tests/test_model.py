"""Pruebas unitarias para el módulo src/model.py."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.model import (
    compare_models,
    generate_submission,
    train_baseline_logistic_regression,
    train_random_forest,
)


@pytest.fixture
def synthetic_dataset():
    """Genera datos sintéticos con señal clara para verificar entrenamiento."""
    np.random.seed(42)
    n = 60
    # Dos variables predictoras simples
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    # Target dependiente de x1 y x2
    prob = 1 / (1 + np.exp(-(x1 * 1.5 + x2 * 0.8)))
    y = (prob > 0.5).astype(int)

    X_train = pd.DataFrame({"feat1": x1, "feat2": x2})
    y_train = pd.Series(y, name="Survived")

    X_test = pd.DataFrame(
        {"feat1": np.random.normal(0, 1, 10), "feat2": np.random.normal(0, 1, 10)}
    )
    passenger_ids = pd.Series(range(100, 110), name="PassengerId")

    return X_train, y_train, X_test, passenger_ids


def test_train_baseline_logistic_regression(synthetic_dataset):
    X_train, y_train, _, _ = synthetic_dataset
    model_lr, metrics = train_baseline_logistic_regression(X_train, y_train, cv_splits=3)

    assert hasattr(model_lr, "predict")
    assert "accuracy_mean" in metrics
    assert 0.0 <= metrics["accuracy_mean"] <= 1.0
    assert metrics["accuracy_mean"] > 0.5  # Mejor que azar


def test_train_random_forest(synthetic_dataset):
    X_train, y_train, _, _ = synthetic_dataset
    model_rf, metrics, importances = train_random_forest(
        X_train, y_train, cv_splits=3, n_estimators=20, max_depth=3, random_state=42
    )

    assert hasattr(model_rf, "predict")
    assert "accuracy_mean" in metrics
    assert isinstance(importances, pd.DataFrame)
    assert set(importances.columns) == {"Feature", "Importance"}
    assert len(importances) == 2


def test_compare_models():
    metrics_a = {
        "accuracy_mean": 0.82,
        "accuracy_std": 0.02,
        "precision_mean": 0.80,
        "recall_mean": 0.75,
        "f1_mean": 0.77,
        "roc_auc_mean": 0.88,
    }
    metrics_b = {
        "accuracy_mean": 0.83,
        "accuracy_std": 0.01,
        "precision_mean": 0.81,
        "recall_mean": 0.76,
        "f1_mean": 0.78,
        "roc_auc_mean": 0.89,
    }
    comp_df = compare_models(metrics_a, metrics_b)
    assert len(comp_df) == 6
    assert "Métrica" in comp_df.columns
    assert "Regresión Logística (Baseline)" in comp_df.columns
    assert "Random Forest" in comp_df.columns


def test_generate_submission(synthetic_dataset, tmp_path):
    X_train, y_train, X_test, passenger_ids = synthetic_dataset
    model_lr, _ = train_baseline_logistic_regression(X_train, y_train, cv_splits=3)

    out_file = tmp_path / "test_sub.csv"
    sub_df = generate_submission(model_lr, X_test, passenger_ids, str(out_file))

    assert out_file.exists()
    assert len(sub_df) == 10
    assert list(sub_df.columns) == ["PassengerId", "Survived"]
    assert (sub_df["PassengerId"] == passenger_ids).all()
