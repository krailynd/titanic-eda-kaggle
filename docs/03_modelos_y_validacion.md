# 🤖 Modelado, Validación Cruzada y Evaluación Comparativa

Este documento analiza el entrenamiento, calibración y comparación formal de los modelos de aprendizaje automático implementados en el proyecto.

---

## 1. Estrategia de Validación: Stratified 5-Fold Cross Validation

Dado el tamaño del conjunto de entrenamiento (891 muestras) y la distribución de clases (38.4% de supervivientes), dividir el dataset en un único corte simple (*train/test split*) generaría una métrica inestable dependiente de la semilla de partición.

Para garantizar rigor metodológico:
* Empleamos **Validación Cruzada Estratificada en 5 Pliegues (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`)**.
* Cada pliegue preserva exactamente la misma proporción de sobrevivientes (38.4%) y fallecidos (61.6%).
* Las métricas reportadas corresponden al **promedio ($\mu$) y desviación estándar ($\sigma$)** a través de las 5 evaluaciones independientes.

---

## 2. Definición de los Estimadores

### Modelo 1: Regresión Logística (Baseline Lineal)
* **Objetivo:** Establecer una cota inferior de rendimiento con un modelo lineal altamente interpretable.
* **Preprocesamiento:** Integrado con `StandardScaler` en un `Pipeline` para normalizar las variables numéricas continuas (`Age`, `Fare`, `FamilySize`).
* **Hiperparámetros:** `max_iter=1000`, `random_state=42`.

### Modelo 2: Random Forest Classifier (Ensamble No Lineal)
* **Objetivo:** Capturar interacciones no lineales entre variables (por ejemplo, el efecto cruzado entre clase, edad y género).
* **Control de Regularización:** Para evitar el sobreajuste propio de árboles no acotados:
  * `n_estimators=100` (100 árboles de decisión en paralelo).
  * `max_depth=5` (profundidad máxima limitada para evitar memorización).
  * `min_samples_split=4` (mínimo de muestras para dividir un nodo).
  * `random_state=42`.

---

## 3. Matriz Comparativa de Resultados

Resultados medidos en la validación cruzada estratificada sobre `train.csv`:

| Métrica de Rendimiento | Regresión Logística (Baseline) | Random Forest Classifier | Análisis Comparativo |
|---|---|---|---|
| **Accuracy (Exactitud Promedio)** | **82.94%** | **82.83%** | Virtualmente idénticos; ambos superan la meta del 80%. |
| **Varianza entre Pliegues (Std $\sigma$)** | **±2.08%** | **±1.00%** | **Random Forest es el doble de estable** entre subconjuntos de datos. |
| **Precision (Precisión en Positivos)** | 78.74% | **80.47%** | Random Forest comete menos falsos positivos al predecir quién sobrevive. |
| **Recall (Sensibilidad)** | **76.01%** | 73.07% | La Regresión Logística recupera ligeramente más supervivientes reales. |
| **F1-Score (Media Armónica)** | **77.34%** | 76.49% | Balance muy parejo entre precisión y exhaustividad. |
| **ROC-AUC (Área bajo la Curva)** | 87.22% | **87.43%** | Excelente capacidad de discriminación probabilística en ambos modelos. |

---

## 4. Importancia de Características (Feature Importance - Gini)

El análisis de impureza de Gini del `RandomForestClassifier` confirma cuantitativamente los hallazgos del análisis exploratorio:

| Posición | Variable | Importancia (%) | Interpretación Conceptual |
|:---:|---|:---:|---|
| **1** | `Title_Mr` | **20.25%** | Penalización severa a varones adultos sin título especial. |
| **2** | `Sex` | **19.54%** | Factor dominante de género biológico en la evacuación. |
| **3** | `Fare` | **10.25%** | Proxy continuo de poder adquisitivo y ubicación de cabina. |
| **4** | `Pclass_3` | **8.71%** | Desventaja estructural de viajar en tercera clase. |
| **5** | `Title_Mrs` | **7.26%** | Prioridad de mujeres casadas con familia a cargo. |
| **6** | `Age` | **6.67%** | Impacto de edad (priorización de infantes). |
| **7** | `FamilySize` | **5.39%** | Ventaja de núcleos pequeños vs solitarios / familias masivas. |
| **8** | `Title_Miss` | **4.91%** | Prioridad de mujeres jóvenes solteras. |
| **9** | `Deck_U` | **2.67%** | Penalización por no poseer cabina registrada de primera/segunda clase. |
| **10** | `Pclass_2` | **2.32%** | Nivel medio socioeconómico. |

---

## 5. Selección del Modelo de Entrega (Submission)

Aunque la Regresión Logística obtuvo un $0.11\%$ más en exactitud media ($82.94\%$ vs $82.83\%$), **seleccionamos Random Forest como modelo oficial de entrega (`submission.csv`)** por tres motivos técnicos:
1. **Menor Varianza ($\pm 1.00\%$ vs $\pm 2.08\%$):** Su dispersión a través de los pliegues es la mitad, lo que predice menor propensión a sorpresas o degradación en el leaderboard oculto de Kaggle.
2. **Mayor Precisión ($80.47\%$ vs $78.74\%$):** Minimiza los falsos supervivientes, siendo más conservador y certero en sus asignaciones.
3. **Distribución de Predicciones Realista:** En el conjunto de prueba (`test.csv` de 418 pasajeros), predijo una tasa de supervivencia del **$38.28\%$** (160 supervivientes y 258 fallecidos), perfectamente alineada con la tasa histórica observada en el desastre ($38.38\%$).
