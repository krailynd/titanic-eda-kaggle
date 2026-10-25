"""Feature engineering y preprocesamiento para el Titanic.

Módulo de funciones reutilizables para análisis exploratorio,
transformación de variables y preparación de datos para modelado.
"""

from typing import List, Tuple
import pandas as pd


def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el título de cortesía (Mr/Mrs/Miss/...) desde la columna Name.

    Maneja títulos compuestos y agrupa categorías poco frecuentes en 'Rare'.
    """
    df = df.copy()
    # Regex robusto que busca la palabra previa al punto en el campo Name
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    # Agrupar títulos nobles, militares o raros en 'Rare'
    rare = [
        "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
        "Rev", "Sir", "Jonkheer", "Dona"
    ]
    df["Title"] = df["Title"].replace(rare, "Rare")

    # Estandarizar variantes francesas y abreviaciones comunes
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Por seguridad, si queda algún valor nulo residual, asignar 'Rare'
    df["Title"] = df["Title"].fillna("Rare")
    return df


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Crea FamilySize = SibSp + Parch + 1 (incluye al propio pasajero)."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


def add_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variable binaria IsAlone (1 si viaja solo, 0 si viaja en familia)."""
    df = df.copy()
    if "FamilySize" not in df.columns:
        df = add_family_size(df)
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def extract_deck(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae la letra de la cabina (Deck) como proxy de ubicación en el barco.

    Si no posee cabina informada, asigna 'U' (Unknown). Reemplaza 'T' (raro) por 'U'.
    """
    df = df.copy()
    deck = df["Cabin"].astype(str).str[0].fillna("U")
    deck = deck.replace({"n": "U", "T": "U"})
    df["Deck"] = deck
    return df


def impute_missing_values(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Imputa valores faltantes en train y test sin data leakage.

    Todas las estadísticas (medianas, modas) se calculan EXCLUSIVAMENTE
    sobre el conjunto de entrenamiento (train_df) y se aplican a ambos.
    """
    train = train_df.copy()
    test = test_df.copy()

    # 1. Asegurar títulos para imputación condicional de edad
    if "Title" not in train.columns:
        train = extract_title(train)
    if "Title" not in test.columns:
        test = extract_title(test)

    # 2. Imputar Age por mediana del grupo (Pclass, Title) calculado en train
    age_medians = train.groupby(["Pclass", "Title"])["Age"].median()
    global_age_median = train["Age"].median()

    def fill_age(row: pd.Series) -> float:
        if pd.notna(row["Age"]):
            return float(row["Age"])
        return float(age_medians.get((row["Pclass"], row["Title"]), global_age_median))

    train["Age"] = train.apply(fill_age, axis=1)
    test["Age"] = test.apply(fill_age, axis=1)

    # 3. Imputar Embarked usando la moda calculada en train ('S')
    embarked_mode = train["Embarked"].mode()[0]
    train["Embarked"] = train["Embarked"].fillna(embarked_mode)
    test["Embarked"] = test["Embarked"].fillna(embarked_mode)

    # 4. Imputar Fare (en test existe 1 pasajero con tarifa nula) por mediana de Pclass en train
    fare_medians = train.groupby("Pclass")["Fare"].median()
    global_fare_median = train["Fare"].median()

    def fill_fare(row: pd.Series) -> float:
        if pd.notna(row["Fare"]):
            return float(row["Fare"])
        return float(fare_medians.get(row["Pclass"], global_fare_median))

    train["Fare"] = train.apply(fill_fare, axis=1)
    test["Fare"] = test.apply(fill_fare, axis=1)

    return train, test


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, include_deck: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """Pipeline completo de transformación, feature engineering y codificación.

    Retorna:
        X_train (pd.DataFrame): Matriz de features de entrenamiento.
        y_train (pd.Series): Vector objetivo (Survived).
        X_test (pd.DataFrame): Matriz de features de prueba alineada.
        feature_names (List[str]): Lista canónica de variables resultantes.
    """
    train = train_df.copy()
    test = test_df.copy()

    # 1. Features derivadas
    train = extract_title(train)
    test = extract_title(test)

    train = add_family_size(train)
    test = add_family_size(test)

    train = add_is_alone(train)
    test = add_is_alone(test)

    if include_deck:
        train = extract_deck(train)
        test = extract_deck(test)

    # 2. Imputación sin data leakage
    train, test = impute_missing_values(train, test)

    # 3. Transformación binaria de Sex (male: 0, female: 1)
    train["Sex"] = train["Sex"].map({"male": 0, "female": 1}).astype(int)
    test["Sex"] = test["Sex"].map({"male": 0, "female": 1}).astype(int)

    # 4. Variables categóricas a codificar vía One-Hot Encoding
    cat_cols = ["Pclass", "Embarked", "Title"]
    if include_deck:
        cat_cols.append("Deck")

    # Concatenar para asegurar que el espacio de columnas dummy sea idéntico
    y_train = train["Survived"].astype(int)
    train_features = train.drop(columns=["Survived"])

    combined = pd.concat([train_features, test], axis=0, ignore_index=True)
    combined_encoded = pd.get_dummies(
        combined, columns=cat_cols, drop_first=True, dtype=int
    )

    # 5. Filtrar columnas crudas que no van al modelo
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    feature_cols = [c for c in combined_encoded.columns if c not in drop_cols]

    X_train = combined_encoded.iloc[: len(train)][feature_cols].copy()
    X_test = combined_encoded.iloc[len(train) :][feature_cols].copy()

    return X_train, y_train, X_test, feature_cols
