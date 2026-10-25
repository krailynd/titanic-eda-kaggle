"""Entrenamiento, evaluación y comparación de modelos para Titanic.

Implementa el baseline (Regresión Logística con escalado) y el modelo
de ensamble (Random Forest), evaluando mediante validación cruzada estratificada.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cross_validate_model(
    model: Any, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5, random_state: int = 42
) -> Dict[str, float]:
    """Evalúa un estimador mediante StratifiedKFold calculando múltiples métricas.

    Retorna un diccionario con los promedios y desviación estándar de:
    - accuracy (media y std)
    - precision
    - recall
    - f1
    - roc_auc
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    accs: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    f1s: List[float] = []
    aucs: List[float] = []

    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y

    for train_idx, val_idx in skf.split(X_arr, y_arr):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        accs.append(accuracy_score(y_val, preds))
        precs.append(precision_score(y_val, preds, zero_division=0))
        recs.append(recall_score(y_val, preds, zero_division=0))
        f1s.append(f1_score(y_val, preds, zero_division=0))

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, probs))
        else:
            aucs.append(float("nan"))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "f1_mean": float(np.mean(f1s)),
        "roc_auc_mean": float(np.mean(aucs)),
    }


def train_baseline_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, cv_splits: int = 5
) -> Tuple[Pipeline, Dict[str, float]]:
    """Entrena y valida el baseline de Regresión Logística.

    Incluye un StandardScaler dentro de un Pipeline para prevenir data leakage
    durante la validación cruzada.
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    metrics = cross_validate_model(pipeline, X_train, y_train, cv_splits=cv_splits)

    # Entrenar sobre todo el conjunto de entrenamiento para la inferencia final
    pipeline.fit(X_train, y_train)
    return pipeline, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = 5,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_split: int = 4,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, float], pd.DataFrame]:
    """Entrena y valida un RandomForestClassifier.

    Retorna el modelo entrenado en todo train, sus métricas de validación
    cruzada y el ranking de importancia de características.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )

    metrics = cross_validate_model(rf, X_train, y_train, cv_splits=cv_splits)

    # Ajuste final
    rf.fit(X_train, y_train)

    feature_importances = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": rf.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return rf, metrics, feature_importances


def compare_models(
    metrics_baseline: Dict[str, float], metrics_rf: Dict[str, float]
) -> pd.DataFrame:
    """Construye una tabla comparativa formateada entre el baseline y Random Forest."""
    data = {
        "Métrica": [
            "Accuracy (CV Promedio)",
            "Desviación Estándar (Std)",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
        ],
        "Regresión Logística (Baseline)": [
            f"{metrics_baseline['accuracy_mean'] * 100:.2f}%",
            f"±{metrics_baseline['accuracy_std'] * 100:.2f}%",
            f"{metrics_baseline['precision_mean'] * 100:.2f}%",
            f"{metrics_baseline['recall_mean'] * 100:.2f}%",
            f"{metrics_baseline['f1_mean'] * 100:.2f}%",
            f"{metrics_baseline['roc_auc_mean'] * 100:.2f}%",
        ],
        "Random Forest": [
            f"{metrics_rf['accuracy_mean'] * 100:.2f}%",
            f"±{metrics_rf['accuracy_std'] * 100:.2f}%",
            f"{metrics_rf['precision_mean'] * 100:.2f}%",
            f"{metrics_rf['recall_mean'] * 100:.2f}%",
            f"{metrics_rf['f1_mean'] * 100:.2f}%",
            f"{metrics_rf['roc_auc_mean'] * 100:.2f}%",
        ],
    }
    return pd.DataFrame(data)


def generate_submission(
    model: Any, X_test: pd.DataFrame, passenger_ids: pd.Series, output_path: str
) -> pd.DataFrame:
    """Genera las predicciones sobre el conjunto de test y exporta el CSV de submission."""
    preds = model.predict(X_test)
    submission_df = pd.DataFrame(
        {
            "PassengerId": passenger_ids.values,
            "Survived": preds.astype(int),
        }
    )
    submission_df.to_csv(output_path, index=False)
    return submission_df
