# 🖥️ Guía de Interpretación de la Salida de Consola (Paso a Paso)

Este documento explica de forma detallada, didáctica e intuitiva **qué significa cada línea, número, bloque y porcentaje** que emite el pipeline (`python src/pipeline.py`) en tu terminal, cómo funciona por dentro y para qué sirve.

---

## 📸 Captura de Ejecución Real en Terminal

![Ejecución del Pipeline en Consola](../assets/pipeline_execution.png)

---

## 🧭 La Gran Pregunta: ¿De qué trata todo esto?

En términos simples:
> **La computadora lee una lista histórica de 891 pasajeros del Titanic donde sabemos quién se salvó y quién no. Aprende las reglas y patrones humanos del naufragio (quiénes tuvieron prioridad, quiénes estaban atrapados, quiénes pagaron más), y luego usa ese conocimiento para predecir el destino de 418 pasajeros nuevos cuyo desenlace nadie conoce.**

A continuación, analizamos **bloque por bloque** lo que viste en tu pantalla al ejecutar el comando.

---

## 1. Bloque `[1/5]`: Carga y Dimensiones de los Datos

```text
[1/5] Cargando datos desde /mnt/ssd-kingston/proyectos/ciencia-datos/titanic-eda-kaggle/data...
      Train: 891 filas, 12 columnas
      Test:  418 filas, 11 columnas
```

### ¿Qué dice cada cosa?
* **`Train: 891 filas, 12 columnas`:**  
  Son **891 pasajeros históricos**. Cada uno tiene 12 datos registrados:
  `PassengerId`, `Survived` (1 = sobrevivió, 0 = falleció), `Pclass` (clase), `Name`, `Sex`, `Age`, `SibSp` (hermanos/cónyuges), `Parch` (padres/hijos), `Ticket`, `Fare` (precio del boleto), `Cabin` (camarote) y `Embarked` (puerto de abordaje).  
  Este archivo es el **material de estudio** del modelo.
* **`Test: 418 filas, 11 columnas`:**  
  Son **418 pasajeros nuevos**. Tienen exactamente los mismos datos que Train, **excepto la columna `Survived`**. ¿Por qué tiene 11 columnas en vez de 12? Porque la respuesta está oculta en los servidores de Kaggle: es el **examen final**.
* **¿Para qué sirve?**  
  Carga los datos en bruto a la memoria para verificar que los archivos no estén vacíos ni corruptos antes de procesarlos.

---

## 2. Bloque `[2/5]`: Feature Engineering e Imputación sin Data Leakage

```text
[2/5] Aplicando Feature Engineering e Imputación sin data leakage...
      Variables finales seleccionadas (22):
      Sex, Age, SibSp, Parch, Fare, FamilySize, IsAlone, Pclass_2, Pclass_3, Embarked_Q, Embarked_S, 
      Title_Miss, Title_Mr, Title_Mrs, Title_Rare, Deck_B, Deck_C, Deck_D, Deck_E, Deck_F, Deck_G, Deck_U
```

### ¿Qué significa cada término?
1. **Imputación:** En los datos originales faltan datos (casilleros vacíos). Por ejemplo, a casi el 20% de los pasajeros les faltaba la edad. En vez de inventar o tirar una media ciega:
   * A cada pasajero sin edad le calculamos la **mediana etaria según su clase y su título social** (un nene `Master` de 3ra clase recibe la edad típica de los nenes de su clase).
2. **Sin Data Leakage (Sin Fuga de Datos):**  
   * *El error de novato:* Calcular el promedio usando los datos de train y test juntos. Eso es "espiar el examen".
   * *Nuestra solución:* Todas las estadísticas se calcularon **estrictamente sobre Train**. Test solo recibió los valores precalculados.
3. **Feature Engineering (Ingeniería de Atributos):**  
   Creamos variables nuevas a partir del texto y los números existentes:
   * **`Title_*`:** Extraído del nombre (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`).
   * **`FamilySize`:** Suma de `SibSp + Parch + 1` (tamaño total de la familia a bordo).
   * **`IsAlone`:** Vale `1` si viajaba solo, `0` si viajaba acompañado.
   * **`Deck_*`:** Letra de la cabina (`A`, `B`, `C`... o `U` de Unknown).
4. **One-Hot Encoding (¿Por qué ahora son 22 variables?):**  
   Las computadoras no entienden palabras como `"Tercera Clase"` o `"Southampton"`. Las variables con categorías se descomponen en interruptores binarios (`0` o `1`):
   * `Pclass_2 = 1` si viajaba en segunda clase, `0` si no.
   * `Pclass_3 = 1` si viajaba en tercera clase, `0` si no. (Si ambos valen `0`, la computadora deduce que viajaba en primera clase).

---

## 3. Bloque `[3/5]`: Modelo 1 — Regresión Logística (Baseline)

```text
[3/5] Entrenando Modelo 1: Regresión Logística (Baseline)...
      Accuracy CV (5-fold): 82.94% (±2.08%)
```

### ¿Qué dice y cómo funciona?
* **Regresión Logística:** Es un modelo lineal clásico de clasificación binaria. Calcula una fórmula matemática de pesos ponderados que suma o resta puntos a la probabilidad de sobrevivir según las variables del pasajero.
* **Baseline (Línea Base):** Es tu punto de referencia mínimo. Significa: *"Cualquier modelo complejo que inventemos después está obligado a justificar su complejidad superando o igualando este 82.94%"*.
* **CV (5-fold) - Validación Cruzada en 5 pliegues:**  
  Dividimos los 891 pasajeros en 5 grupos iguales de ~178 personas. Entrenamos con 4 grupos y evaluamos con el 5to, rotando 5 veces. Así evitamos tener suerte con una sola partición.
* **`82.94% (±2.08%)`:**  
  * **82.94%:** De cada 100 pasajeros, el modelo predijo correctamente el destino de 83.
  * **`±2.08%`:** Es la **desviación estándar**. Significa que en algunos pliegues sacó 85% y en otros cayó a 80.8%.

---

## 4. Bloque `[4/5]`: Modelo 2 — Random Forest Classifier

```text
[4/5] Entrenando Modelo 2: Random Forest Classifier...
      Accuracy CV (5-fold): 82.83% (±1.00%)
```

### ¿Cómo funciona Random Forest?
No utiliza una sola fórmula matemática. Crea un **"bosque" de 100 árboles de decisión diferentes**. Cada árbol analiza preguntas sucesivas:
> *¿Es varón adulto? Si es sí $\rightarrow$ ¿Viaja en 3ra clase? Si es sí $\rightarrow$ ¿Pagó menos de £10? $\rightarrow$ Predicción: Fallece.*

Al final, los 100 árboles votan democráticamente y gana la mayoría.

### 🔍 La clave del número `±1.00%`:
Fijate que la exactitud media es casi idéntica a la Regresión Logística (82.83% vs 82.94%), **pero la dispersión es la mitad ($\pm 1.00\%$ contra $\pm 2.08\%$)**.  
Esto demuestra que el Random Forest es **el doble de consistente y estable**. No sufre de altibajos pronunciados según los datos que le toquen.

---

## 5. El Ranking: ¿Qué variables decidieron la vida o la muerte?

```text
Top 5 Variables Más Importantes (Random Forest):
      1. Title_Mr             -> 20.25%
      2. Sex                  -> 19.54%
      3. Fare                 -> 10.25%
      4. Pclass_3             -> 8.71%
      5. Title_Mrs            -> 7.26%
```

El modelo aprendió de forma autónoma los patrones sociológicos reales de la catástrofe:
1. **`Title_Mr` (20.25%):** Ser un varón adulto sin título noble fue el factor individual más castigado en la evacuación.
2. **`Sex` (19.54%):** La regla estricta de *"mujeres y niños primero"*. Las mujeres tuvieron una tasa de supervivencia del **74.2%** vs **18.9%** en varones.
3. **`Fare` (10.25%):** Quien pagó más dinero tenía camarotes en cubiertas superiores, a metros de los botes.
4. **`Pclass_3` (8.71%):** Los pasajeros de 3ra clase estaban en los compartimentos más profundos del barco y sufrieron la mayor tasa de mortalidad.
5. **`Title_Mrs` (7.26%):** Las mujeres casadas con familia a cargo recibieron máxima prioridad en el abordaje.

---

## 6. Bloque `[5/5]`: Comparación Formal de Rendimiento

```text
[5/5] Comparación Formal de Rendimiento:
                  Métrica Regresión Logística (Baseline) Random Forest
   Accuracy (CV Promedio)                         82.94%        82.83%
Desviación Estándar (Std)                         ±2.08%        ±1.00%
                Precision                         78.74%        80.47%
                   Recall                         76.01%        73.07%
                 F1-Score                         77.34%        76.49%
                  ROC-AUC                         87.22%        87.43%
```

### ¿Qué significa cada métrica en castellano?

* **Accuracy (Exactitud General):**  
  Porcentaje de aciertos totales sobre el total de casos. Ambos modelos rondan el 83%.
* **Desviación Estándar (Std):**  
  Mide la inestabilidad del modelo entre distintos subconjuntos de datos. **Gana Random Forest (1.00%)**, siendo mucho más robusto.
* **Precision (80.47% en Random Forest):**  
  De todas las personas que el modelo predijo que *sobrevivirían*, **¿cuántas realmente sobrevivieron?**  
  Un 80.47% significa que el modelo comete muy pocos falsos positivos (no manda a vivir a gente que murió).
* **Recall / Sensibilidad (76.01% en Regresión Logística):**  
  De todas las personas que *realmente sobrevivieron en la historia*, ¿a cuántas logró rescatar el modelo?  
  La Regresión Logística atrapa un 76%, mientras Random Forest un 73%.
* **F1-Score (77.34% vs 76.49%):**  
  La media armónica entre Precision y Recall. Es la métrica reina cuando las clases no están perfectamente equilibradas.
* **ROC-AUC (87.43% en Random Forest):**  
  Mide qué tan bueno es el modelo ordenando a los pasajeros por probabilidad de supervivencia de mayor a menor. Arriba de 85% se considera un clasificador de alto rendimiento.

---

## 7. Archivos de Salida (`Submissions`)

```text
Submissions exportadas con éxito:
      - /mnt/ssd-kingston/proyectos/ciencia-datos/titanic-eda-kaggle/data/submission_lr.csv
      - /mnt/ssd-kingston/proyectos/ciencia-datos/titanic-eda-kaggle/data/submission_rf.csv
      - /mnt/ssd-kingston/proyectos/ciencia-datos/titanic-eda-kaggle/data/submission.csv (Candidato oficial a Kaggle)
```

Cada archivo tiene exactamente **418 filas** con dos columnas:
```csv
PassengerId,Survived
892,0
893,0
894,0
...
1309,1
```

* **`submission.csv`** es el archivo oficial que subís a Kaggle.
* Contiene el veredicto del **Random Forest**, que predice que sobrevivió el **38.28%** de las personas en test (un reflejo idéntico del **38.38%** de supervivencia real del Titanic histórico).
