# 🚢 Titanic EDA & Machine Learning — Kaggle Competition

> **Primer entregable de mi roadmap hacia Data Science & AI Engineering.**  
> Flujo profesional completo *end-to-end*: **datos crudos $\rightarrow$ limpieza sin data leakage $\rightarrow$ EDA riguroso $\rightarrow$ feature engineering $\rightarrow$ baseline vs ensamble $\rightarrow$ validación cruzada estratificada $\rightarrow$ submission verificada $\rightarrow$ documentación técnica.**

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Titanic_Machine_Learning_from_Disaster-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/titanic)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-v2.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Tests](https://img.shields.io/badge/Tests-10%20Passed%20(pytest)-brightgreen?logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Planteamiento del Problema

Predecir la supervivencia de los pasajeros del naufragio del RMS Titanic a partir de atributos sociodemográficos, económicos y de viaje (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Cabin`, etc.).

* **Dataset:** 891 registros de entrenamiento con etiqueta (`train.csv`) y 418 registros de evaluación a ciegas (`test.csv`).
* **Métrica Objetivo:** Accuracy honesto en Validación Cruzada Estratificada $> \mathbf{78\% - 82\%}$.
* **Filosofía del Proyecto:** Rigor metodológico sin copiar soluciones ni incurrir en sobreajuste por memorización de listas públicas de Wikipedia.

---

## 📊 Resultados Empíricos Observados

### 📸 Ejecución del Pipeline en Consola
![Ejecución del Pipeline en Consola](assets/pipeline_execution.png)

*(Para la explicación didáctica línea por línea de esta salida, ver [docs/06_interpretacion_salida_consola.md](docs/06_interpretacion_salida_consola.md). Para el desglose riguroso de fórmulas y optimización, ver [docs/07_fundamentos_matematicos.md](docs/07_fundamentos_matematicos.md)).*

Evaluación realizada mediante **Stratified 5-Fold Cross Validation** sobre el conjunto de entrenamiento:

| Modelo | Accuracy ($\mu \pm \sigma$) | Precision | Recall | F1-Score | ROC-AUC | Estado |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Regresión Logística (Baseline)** | **82.94%** $\pm$ 2.08% | 78.74% | **76.01%** | **77.34%** | 87.22% | Evaluado |
| **Random Forest Classifier (Ensamble)** | **82.83%** $\pm$ **1.00%** | **80.47%** | 73.07% | 76.49% | **87.43%** | **Seleccionado (Submission Oficial)** |

### 🏆 ¿Por qué Random Forest para la entrega oficial?
Aunque la Regresión Logística obtuvo un $+0.11\%$ en exactitud promedio, **Random Forest redujo la varianza entre pliegues a la mitad ($\pm 1.00\%$ vs $\pm 2.08\%$)** y demostró una mayor **precisión (80.47%)**, garantizando mayor consistencia frente a datos no observados.

---

## 💡 Hallazgos Principales del EDA e Ingeniería de Atributos

1. **Protocolo "Mujeres y Niños Primero":** La tasa de supervivencia femenina fue de **74.20%** frente a apenas **18.89%** en varones.
2. **Estratificación Social:** Primera Clase alcanzó un **62.96%** de supervivencia (£84 de tarifa promedio), frente al **24.24%** de Tercera Clase (£13.68 de tarifa promedio).
3. **Poder Predictivo de los Títulos (`Title`):** Al extraer el título de cortesía desde el nombre, los niños varones identificados como `Master` sobrevivieron un **57.50%**, diferenciándose drásticamente de los adultos `Mr` (**15.67%**).
4. **Zona Óptima Familiar (`FamilySize`):** Núcleos familiares de 2 a 4 integrantes lograron entre **55% y 72%** de supervivencia, superando con creces a viajeros solos (~30%) y a familias numerosas $\ge 5$ miembros ($<20\%$).

### Top 5 Características Más Importantes (Random Forest):
1. `Title_Mr` $\rightarrow$ **20.25%**
2. `Sex` $\rightarrow$ **19.54%**
3. `Fare` $\rightarrow$ **10.25%**
4. `Pclass_3` $\rightarrow$ **8.71%**
5. `Title_Mrs` $\rightarrow$ **7.26%**

---

## 🗂️ Estructura del Repositorio

```
titanic-eda-kaggle/
├── assets/
│   └── pipeline_execution.png  # Captura de ejecución real en terminal
├── data/
│   ├── README.md               # Instrucciones de descarga de Kaggle
│   ├── train.csv               # 891 registros (ignorado en git)
│   ├── test.csv                # 418 registros (ignorado en git)
│   ├── gender_submission.csv   # Benchmark de género
│   ├── submission.csv          # Submission oficial generada (Random Forest)
│   ├── submission_lr.csv       # Submission generada (Regresión Logística)
│   └── submission_rf.csv       # Submission generada (Random Forest)
├── docs/                       # Documentación técnica exhaustiva
│   ├── 01_arquitectura_y_pipeline.md
│   ├── 02_eda_hallazgos_y_patrones.md
│   ├── 03_modelos_y_validacion.md
│   ├── 04_guia_ejecucion_y_reproducibilidad.md
│   ├── 05_aprendizajes_tecnica_feynman.md
│   ├── 06_interpretacion_salida_consola.md
│   └── 07_fundamentos_matematicos.md
├── notebooks/
│   └── 01_eda_titanic.ipynb    # Notebook ejecutable con análisis visual
├── src/                        # Código modular de producción
│   ├── __init__.py
│   ├── features.py             # Feature engineering e imputación sin data leakage
│   ├── model.py                # Entrenamiento, evaluación CV y exportación
│   └── pipeline.py             # CLI runner end-to-end
├── tests/                      # Suite de pruebas automatizadas
│   ├── __init__.py
│   ├── test_features.py        # Pruebas de transformación de features
│   └── test_model.py           # Pruebas de modelos y generación de submission
├── pytest.ini                  # Configuración de pytest
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Este documento
```

---

## 🚀 Guía de Reproducción Rápida

### 1. Clonar e Instalar
```bash
git clone https://github.com/krailynd/titanic-eda-kaggle.git
cd titanic-eda-kaggle

# Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
pip install pytest
```

### 2. Descargar Datos
Colocar `train.csv` y `test.csv` desde [Kaggle Titanic Data](https://www.kaggle.com/competitions/titanic/data) dentro del directorio `data/`.

### 3. Ejecutar Pipeline Completo vía CLI
```bash
python src/pipeline.py
```

### 4. Ejecutar Pruebas Automatizadas
```bash
pytest -v
```

### 5. Abrir Notebook Interactivo
```bash
jupyter notebook notebooks/01_eda_titanic.ipynb
```

---

## 🪜 Checklist de Trabajo del Roadmap

- [x] **Setup:** Repositorio, `.gitignore`, dependencias y datasets en `data/`.
- [x] **Carga y Diagnóstico:** Inspección dimensional y cuantificación de nulos.
- [x] **EDA Completo:** Gráficos de barras, distribuciones KDE, boxplots y correlaciones con Seaborn.
- [x] **Feature Engineering Modular:** Extracción de `Title`, cálculo de `FamilySize`, `IsAlone`, `Deck` en `src/features.py`.
- [x] **Imputación sin Data Leakage:** Imputación de edad por `(Pclass, Title)` calculada estrictamente en train.
- [x] **One-Hot Encoding:** Codificación consistente entre train y test.
- [x] **Baseline Lineal:** Regresión Logística con `StandardScaler` en `Pipeline` (82.94% CV).
- [x] **Modelo Ensamble:** Random Forest con regularización de profundidad (82.83% CV, $\sigma=1.00\%$).
- [x] **Generación de Submissions:** Creación de `data/submission.csv` listo para subir a Kaggle.
- [x] **Suite de Tests:** 10 pruebas unitarias con `pytest` en verde.
- [x] **Documentación Técnica:** Manuales de arquitectura, EDA, modelos y técnica Feynman en `docs/`.

---

## 📚 Documentación Técnica Detallada

El proyecto cuenta con 7 manuales exhaustivos que cubren desde la arquitectura hasta las fórmulas matemáticas:

1. [**01. Arquitectura y Pipeline End-to-End**](docs/01_arquitectura_y_pipeline.md): Diseño modular, flujo de datos y aislamiento estricto contra Data Leakage.
2. [**02. Hallazgos y Patrones del EDA**](docs/02_eda_hallazgos_y_patrones.md): Desglose estadístico y sociológico del naufragio (género, clase, familias, puertos).
3. [**03. Modelos, Validación y Métricas**](docs/03_modelos_y_validacion.md): Comparación formal de Regresión Logística vs Random Forest e importancia de variables.
4. [**04. Guía de Ejecución y Reproducibilidad**](docs/04_guia_ejecucion_y_reproducibilidad.md): Manual paso a paso para correr CLI, Jupyter y Pytest.
5. [**05. Aprendizajes Clave (Técnica Feynman)**](docs/05_aprendizajes_tecnica_feynman.md): Explicación conceptual en lenguaje llano sin tecnicismos vacíos.
6. [**06. Interpretación de Salida de Consola (Línea por Línea)**](docs/06_interpretacion_salida_consola.md): Análisis detallado de cada bloque, número y porcentaje del output de terminal con captura real.
7. [**07. Fundamentos Matemáticos y Algorítmicos**](docs/07_fundamentos_matematicos.md): Fórmulas analíticas (Gini, Logit, Sigmoide, Descenso de Gradiente, Reducción de Varianza en Ensambles, F1 y ROC-AUC).

---

## 🧠 Aprendizajes Clave (Técnica Feynman)

1. **El Feature Engineering vence a la complejidad bruta:** Transformar variables basándose en el contexto del problema (`Title`, `FamilySize`) aportó mucho más valor predictivo que ajustar modelos excesivamente complejos.
2. **Prevenir el Data Leakage es un principio inviolable:** Todas las estadísticas (medianas, modas, escaladores) deben aprenderse única y exclusivamente en `train` para evitar métricas infladas artificialmente.
3. **El valor de la baja varianza:** Entre dos modelos con exactitud promedio similar, el que exhibe menor dispersión entre pliegues (`Random Forest` con $\pm 1.00\%$) es el más confiable y seguro para producción.

*Para un desglose conceptual detallado sin tecnicismos vacíos, consultar [docs/05_aprendizajes_tecnica_feynman.md](docs/05_aprendizajes_tecnica_feynman.md).*

---

**Autor:** [Elvis Carbajal](https://elvis-carbajal.vercel.app) — *Construyendo en público hacia Data Science & AI Engineering.*
