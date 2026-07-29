# 🚢 Titanic EDA — Mi Primer Proyecto de Ciencia de Datos

> Primer entregable de mi roadmap autodidacta hacia Data Science & AI Engineering.
> Flujo completo end-to-end: **datos crudos → limpieza → EDA → feature engineering → modelo → evaluación → submission real en Kaggle**.

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/titanic)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 📌 El problema

Predecir qué pasajeros del Titanic sobrevivieron según variables como edad, sexo, clase y puerto de embarque.
Es la competencia introductoria de [Kaggle](https://www.kaggle.com/competitions/titanic) — miles de notebooks públicos para aprender **(sin copiar)**.

**Métrica objetivo:** Accuracy > 78-80% (resultado honesto para un modelo simple; el top del leaderboard ronda 100% con overfitting sospechoso — no es la meta).

## 🎯 Qué busco aprender con este proyecto

- El flujo real de un proyecto de datos, no solo entrenar un modelo
- Pandas para limpieza y exploración (EDA)
- Feature engineering con criterio (títulos del nombre, tamaño familiar)
- Comparar un baseline simple (Regresión Logística) vs un modelo más fuerte (Random Forest)
- Evaluar con honestidad y documentar el proceso

## 🗂️ Estructura

```
titanic-eda-kaggle/
├── data/                  # NO versionado — descargar de Kaggle
│   └── README.md          # Instrucciones de descarga
├── notebooks/
│   └── 01_eda_titanic.ipynb   # Notebook principal (EDA → modelo)
├── src/
│   └── features.py        # Funciones de feature engineering
├── requirements.txt
└── README.md
```

## 🚀 Cómo reproducirlo

1. Clona el repo:
   ```bash
   git clone https://github.com/krailynd/titanic-eda-kaggle.git
   cd titanic-eda-kaggle
   ```
2. Descarga `train.csv` y `test.csv` desde la [competencia de Kaggle](https://www.kaggle.com/competitions/titanic/data) y colócalos en `data/`
3. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Abre el notebook:
   ```bash
   jupyter notebook notebooks/01_eda_titanic.ipynb
   ```
   (También puedes subirlo directo a Kaggle Notebooks)

## 🪜 Plan de trabajo

- [x] Setup: repo + competencia de Kaggle
- [ ] Carga de datos y primer vistazo (shape, tipos, nulos)
- [ ] EDA: nulos en `Age`/`Cabin`, correlaciones con supervivencia
- [ ] Feature engineering: `Title` desde el nombre, `FamilySize = SibSp + Parch`
- [ ] Encoding de categóricas (`Sex`, `Embarked`)
- [ ] Baseline: Regresión Logística
- [ ] Random Forest + comparación
- [ ] Submission a Kaggle
- [ ] Documentar resultados y aprendizajes aquí

## 📊 Resultados

*(Se completará tras la primera submission)*

| Modelo | Accuracy (validación) | Accuracy (Kaggle) |
|---|---|---|
| Regresión Logística (baseline) | — | — |
| Random Forest | — | — |

## 🧠 Aprendizajes clave

*(Se completará al cerrar el proyecto — técnica Feynman: explicar con mis palabras)*

---

**Parte de mi camino público hacia Data Science & AI** — más contexto en [elvis-carbajal.vercel.app](https://elvis-carbajal.vercel.app)
