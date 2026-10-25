"""Pruebas unitarias para el módulo src/features.py."""

import numpy as np
import pandas as pd
import pytest
from src.features import (
    add_family_size,
    add_is_alone,
    extract_deck,
    extract_title,
    impute_missing_values,
    prepare_features,
)


@pytest.fixture
def sample_raw_data():
    """Genera un mini conjunto de datos representativo para pruebas."""
    train_data = {
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [0, 1, 1, 1, 0],
        "Pclass": [3, 1, 3, 1, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)",
            "Moran, Mr. James",
        ],
        "Sex": ["male", "female", "female", "female", "male"],
        "Age": [22.0, 38.0, 26.0, 33.0, np.nan],
        "SibSp": [1, 1, 0, 0, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "330877"],
        "Fare": [7.25, 71.2833, 7.925, 53.1, 8.4583],
        "Cabin": [np.nan, "C85", np.nan, "B77", np.nan],
        "Embarked": ["S", "C", "S", "S", "Q"],
    }

    test_data = {
        "PassengerId": [6, 7],
        "Pclass": [3, 1],
        "Name": ["Kelly, Mr. James", "Oliva y Ocana, Dona. Fermina"],
        "Sex": ["male", "female"],
        "Age": [34.5, 39.0],
        "SibSp": [0, 0],
        "Parch": [0, 0],
        "Ticket": ["330911", "PC 17758"],
        "Fare": [7.8292, np.nan],
        "Cabin": [np.nan, "C105"],
        "Embarked": ["Q", "C"],
    }
    return pd.DataFrame(train_data), pd.DataFrame(test_data)


def test_extract_title(sample_raw_data):
    train, test = sample_raw_data
    train_titles = extract_title(train)["Title"].tolist()
    test_titles = extract_title(test)["Title"].tolist()

    assert train_titles == ["Mr", "Mrs", "Miss", "Rare", "Mr"]
    assert test_titles == ["Mr", "Rare"]  # Dona mapea a Rare


def test_add_family_size(sample_raw_data):
    train, _ = sample_raw_data
    df = add_family_size(train)
    # SibSp + Parch + 1
    assert df["FamilySize"].tolist() == [2, 2, 1, 1, 1]


def test_add_is_alone(sample_raw_data):
    train, _ = sample_raw_data
    df = add_is_alone(train)
    assert df["IsAlone"].tolist() == [0, 0, 1, 1, 1]


def test_extract_deck(sample_raw_data):
    train, test = sample_raw_data
    df_train = extract_deck(train)
    df_test = extract_deck(test)

    assert df_train["Deck"].tolist() == ["U", "C", "U", "B", "U"]
    assert df_test["Deck"].tolist() == ["U", "C"]


def test_impute_missing_values(sample_raw_data):
    train, test = sample_raw_data
    tr_imp, te_imp = impute_missing_values(train, test)

    # En train la fila 5 tenía Age NaN
    assert tr_imp["Age"].isnull().sum() == 0
    # En test la fila 2 tenía Fare NaN
    assert te_imp["Fare"].isnull().sum() == 0


def test_prepare_features(sample_raw_data):
    train, test = sample_raw_data
    X_train, y_train, X_test, feature_names = prepare_features(train, test)

    assert len(X_train) == 5
    assert len(X_test) == 2
    assert len(y_train) == 5
    assert list(X_train.columns) == list(X_test.columns)
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    assert "Sex" in feature_names
    assert "Fare" in feature_names
    assert "Age" in feature_names
