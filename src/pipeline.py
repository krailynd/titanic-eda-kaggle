"""Pipeline CLI ejecutable end-to-end para Titanic EDA Kaggle.

Carga datos crudos -> Aplica feature engineering -> Entrena modelos ->
Imprime métricas comparativas -> Exporta submissions a data/
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Asegurar importación de módulos locales
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from features import prepare_features
from model import (
    compare_models,
    generate_submission,
    train_baseline_logistic_regression,
    train_random_forest,
)


def run_pipeline(data_dir: str = "../data") -> None:
    data_path = Path(data_dir).resolve()
    train_file = data_path / "train.csv"
    test_file = data_path / "test.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"No se encontraron train.csv o test.csv en {data_path}."
        )

    print("=" * 60)
    print("🚢 EJECUTANDO PIPELINE TITANIC EDA KAGGLE")
    print("=" * 60)

    print(f"\n[1/5] Cargando datos desde {data_path}...")
    train_raw = pd.read_csv(train_file)
    test_raw = pd.read_csv(test_file)
    print(f"      Train: {train_raw.shape[0]} filas, {train_raw.shape[1]} columnas")
    print(f"      Test:  {test_raw.shape[0]} filas, {test_raw.shape[1]} columnas")

    print("\n[2/5] Aplicando Feature Engineering e Imputación sin data leakage...")
    X_train, y_train, X_test, feature_names = prepare_features(train_raw, test_raw)
    print(f"      Variables finales seleccionadas ({len(feature_names)}):")
    print(f"      {', '.join(feature_names)}")

    print("\n[3/5] Entrenando Modelo 1: Regresión Logística (Baseline)...")
    lr_model, lr_metrics = train_baseline_logistic_regression(X_train, y_train, cv_splits=5)
    print(
        f"      Accuracy CV (5-fold): {lr_metrics['accuracy_mean']*100:.2f}% "
        f"(±{lr_metrics['accuracy_std']*100:.2f}%)"
    )

    print("\n[4/5] Entrenando Modelo 2: Random Forest Classifier...")
    rf_model, rf_metrics, importances = train_random_forest(
        X_train, y_train, cv_splits=5, n_estimators=100, max_depth=5, min_samples_split=4
    )
    print(
        f"      Accuracy CV (5-fold): {rf_metrics['accuracy_mean']*100:.2f}% "
        f"(±{rf_metrics['accuracy_std']*100:.2f}%)"
    )

    print("\nTop 5 Variables Más Importantes (Random Forest):")
    for idx, row in importances.head(5).iterrows():
        print(f"      {idx+1}. {row['Feature']:<20} -> {row['Importance']*100:.2f}%")

    print("\n[5/5] Comparación Formal de Rendimiento:")
    comparison_df = compare_models(lr_metrics, rf_metrics)
    print(comparison_df.to_string(index=False))

    # Exportar submissions
    sub_lr_path = data_path / "submission_lr.csv"
    sub_rf_path = data_path / "submission_rf.csv"
    sub_default_path = data_path / "submission.csv"

    generate_submission(lr_model, X_test, test_raw["PassengerId"], str(sub_lr_path))
    generate_submission(rf_model, X_test, test_raw["PassengerId"], str(sub_rf_path))
    # El modelo por defecto para entrega es Random Forest por su menor varianza
    generate_submission(rf_model, X_test, test_raw["PassengerId"], str(sub_default_path))

    print(f"\nSubmissions exportadas con éxito:")
    print(f"      - {sub_lr_path}")
    print(f"      - {sub_rf_path}")
    print(f"      - {sub_default_path} (Candidato oficial a Kaggle)")
    print("=" * 60)
    print("PIPELINE COMPLETADO CON ÉXITO")
    print("=" * 60)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent / "data"
    run_pipeline(str(base_dir))
