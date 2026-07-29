"""Feature engineering para el Titanic.

Funciones reutilizables para el notebook 01_eda_titanic.ipynb
"""

import pandas as pd


def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el título (Mr/Mrs/Miss/...) desde la columna Name."""
    df = df.copy()
    df["Title"] = df["Name"].str.extract(", ([A-Za-z]+)\.", expand=False)
    # Agrupar títulos raros en "Rare"
    rare = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
            "Rev", "Sir", "Jonkheer", "Dona"]
    df["Title"] = df["Title"].replace(rare, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    return df


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Crea FamilySize = SibSp + Parch + 1 (incluye al propio pasajero)."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df
