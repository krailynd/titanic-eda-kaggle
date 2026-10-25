# 🏗️ Arquitectura del Sistema y Pipeline End-to-End

Este documento detalla la ingeniería y el flujo de procesamiento de datos desarrollado en el proyecto **Titanic EDA Kaggle**.

---

## 1. El Problema de Negocio y Ciencia de Datos

El naufragio del RMS Titanic (15 de abril de 1912) es uno de los desastres marítimos más estudiados de la historia. De las 2,224 personas a bordo entre pasajeros y tripulación, murieron más de 1,500.

En términos de machine learning, se plantea como un problema de **clasificación binaria supervisada**:
* **Variable objetivo ($y$):** `Survived` $\in \{0, 1\}$ (0 = Falleció, 1 = Sobrevivió).
* **Entradas ($X$):** Atributos sociodemográficos, económicos y de logística del viaje (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Cabin`, etc.).
* **Métrica de éxito:** Exactitud de clasificación (**Accuracy**), complementada con **Precision**, **Recall**, **F1-Score** y **ROC-AUC**.
* **Objetivo:** Alcanzar un desempeño honesto entre **78% y 83%** en validación cruzada, evitando el sobreajuste al ruido o el uso de información filtrada.

---

## 2. Flujo Arquitectónico (Pipeline End-to-End)

El diseño del proyecto sigue una arquitectura desacoplada donde el código de producción vive en módulos testeables (`src/`), consumibles tanto desde la interfaz analítica e interactiva (`notebooks/`) como desde la automatización por terminal (`pipeline.py`).

```
                              ┌────────────────────────┐
                              │      Datos Crudos      │
                              │ (train.csv / test.csv) │
                              └───────────┬────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │   src/features.py                   │
                       │   - extract_title (Regex cortesía)  │
                       │   - add_family_size (SibSp + Parch) │
                       │   - add_is_alone (Binaria)          │
                       │   - extract_deck (Proxy de cabina)  │
                       │   - impute_missing_values (No leak) │
                       │   - One-Hot Encoding (pd.get_dummies│
                       └──────────────────┬──────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             ┌─────────────────────┐             ┌─────────────────────┐
             │ X_train (891, 22)   │             │ X_test (418, 22)    │
             │ y_train (891,)      │             │                     │
             └──────────┬──────────┘             └──────────┬──────────┘
                        │                                   │
                        ▼                                   │
       ┌─────────────────────────────────┐                  │
       │   src/model.py                  │                  │
       │   - StratifiedKFold (5 splits)  │                  │
       │                                 │                  │
       │  [Modelo 1: Regresión Logística]│                  │
       │   StandardScaler + LogisticReg  │                  │
       │   -> CV Accuracy: 82.94%        │                  │
       │                                 │                  │
       │  [Modelo 2: Random Forest]      │                  │
       │   n_est=100, max_depth=5        │                  │
       │   -> CV Accuracy: 82.83%        │                  │
       └────────────────┬────────────────┘                  │
                        │                                   │
                        │        Inferencia / Predicción    │
                        ├───────────────────────────────────┘
                        │
                        ▼
       ┌─────────────────────────────────┐
       │     data/submission.csv         │
       │     (PassengerId, Survived)     │
       │     Listo para Kaggle           │
       └─────────────────────────────────┘
```

---

## 3. Principio de Aislamiento y Prevención de Data Leakage

Uno de los errores más comunes en la ciencia de datos principiante es calcular estadísticas globales (medias, modas o escalas) sobre la unión indiscriminada de entrenamiento y prueba antes de dividir. Esto provoca **fuga de datos (Data Leakage)**: el modelo aprende información de la distribución del conjunto de evaluación que no tendría en un escenario real de despliegue.

### Reglas aplicadas en este pipeline:
1. **Imputación Estratificada Condicional:**
   * La variable `Age` se imputa utilizando la **mediana por `(Pclass, Title)`** calculada **únicamente sobre `train.csv`**.
   * Si en `test.csv` aparece una combinación no vista, se recurre a la mediana global observada en `train.csv`.
2. **Imputación de Categóricas y Tarifas:**
   * La moda de `Embarked` (`'S'`) se extrae de `train.csv`.
   * El pasajero con tarifa nula en `test.csv` se imputa con la mediana de su clase calculada en `train.csv`.
3. **Escalado en Baseline:**
   * En la Regresión Logística, `StandardScaler` se encapsula dentro de un `sklearn.pipeline.Pipeline`, garantizando que durante cada pliegue de la validación cruzada el escalador solo se ajuste (*fit*) sobre el conjunto de entrenamiento local del pliegue.

---

## 4. Estructura Modular del Código

| Módulo | Responsabilidad |
|---|---|
| `src/features.py` | Funciones puras de transformación, ingeniería de atributos, imputación y generación de matrices de diseño compatibles. |
| `src/model.py` | Entrenamiento de estimadores, evaluación mediante validación cruzada estratificada (5-Fold), cálculo de métricas de diagnóstico y exportación de archivos de entrega. |
| `src/pipeline.py` | Punto de entrada CLI para ejecutar el ciclo de vida completo de punta a punta en un solo comando. |
| `notebooks/01_eda_titanic.ipynb` | Cuaderno de investigación con visualizaciones exploratorias (Seaborn), tablas de contingencia y análisis narrativo. |
| `tests/` | Suite automatizada con `pytest` que verifica la consistencia matemática, forma de tensores y ausencia de nulos. |
