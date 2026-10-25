# 🧠 Aprendizajes Clave — Técnica Feynman

Este documento recopila las lecciones fundamentales aprendidas durante el desarrollo del proyecto, explicadas en **lenguaje llano, directo y accesible** (Técnica Feynman: si no podés explicarlo de forma simple, es porque no lo entendiste lo suficiente).

---

## 1. ¿Qué es el Data Leakage (Fuga de Datos) y por qué destruye proyectos en silencio?

> **Analogía cotidiana:** Imaginate que vas a rendir un examen de matemáticas y el profesor te deja practicar con un simulacro, pero por descuido se olvidó las respuestas del examen real anotadas en el margen de la hoja. En la práctica vas a sacar un 10 brillante, pero el día del examen real en la facultad te vas a sacar un 2 porque nunca aprendiste a razonar los problemas desde cero.

En ciencia de datos, el **Data Leakage** ocurre cuando metés información del futuro o del conjunto de prueba dentro del entrenamiento.

### ¿Cómo lo evitamos en este proyecto?
Teníamos que adivinar las edades de los pasajeros que no tenían el dato anotado. La tentación de un novato es calcular la edad promedio de todos los datos juntos (train + test).
Nosotros **calculamos las edades medianas por clase y título ÚNICAMENTE con los pasajeros de entrenamiento (`train.csv`)**. Luego, usamos esa tabla de reglas aprendidas para rellenar a los pasajeros de prueba. Así garantizamos que el modelo no tenga ninguna "ayudita secreta".

---

## 2. El Feature Engineering supera a la complejidad del algoritmo

> **La idea central:** No necesitás una red neuronal de mil capas para resolver un problema tabular. Necesitás que los datos que le das al modelo tengan **sentido humano y contextual**.

Cuando el modelo lee `"Braund, Mr. Owen Harris"` o `"Heikkinen, Miss. Laina"`, para la computadora son solo letras sin valor. Pero cuando le extrajimos el título:
* Descubrió que los nenes varones tenían el título `Master` y sobrevivían más del **57%** de las veces, mientras que los hombres adultos (`Mr`) sobrevivían apenas el **15%**.
* Descubrió que una `Mrs` (mujer casada) tenía casi un **80%** de supervivencia frente al **70%** de una `Miss` soltera.
* Al juntar hermanos, cónyuges, padres e hijos en `FamilySize`, el modelo aprendió que las familias de 2 a 4 personas se cuidaban entre sí y conseguían subir juntos a los botes, mientras que las personas que viajaban solas o las familias gigantes de 7 u 8 integrantes tenían el peor pronóstico.

**Conclusión:** Un buen feature engineering multiplica el rendimiento de cualquier algoritmo sin necesidad de cambiar una sola línea de código del modelo.

---

## 3. ¿Por qué siempre tenés que empezar con un Baseline simple?

> **La trampa del principiante:** Empezar cualquier problema importando XGBoost, CatBoost o redes neuronales profundas sin saber cuál es el piso mínimo aceptable.

En este proyecto arrancamos con una **Regresión Logística estandarizada**:
* Es rápida, no consume memoria y se puede explicar en una servilleta: calcula una línea recta de decisión sumando los pesos de cada variable.
* Nos dio un impresionante **82.94% de exactitud**.
* Nos sirvió de juez: cualquier modelo más pesado o complejo que quisiéramos probar después estaba obligado a superar ese número o a ofrecer ventajas tangibles de estabilidad.

---

## 4. Sesgo vs Varianza en la vida real: ¿Por qué elegimos Random Forest?

Si miramos la foto rápida de la exactitud:
* **Regresión Logística:** 82.94%
* **Random Forest:** 82.83%

Cualquiera diría: *"Gana la Regresión Logística por un 0.11%"*. Pero en ciencia de datos profesional **la media no te cuenta toda la película; la desviación estándar sí**.

* La Regresión Logística tuvo una dispersión de **$\pm 2.08\%$** entre pliegues. En un pliegue sacaba 85% y en otro caía a 80%.
* El Random Forest tuvo una dispersión de **$\pm 1.00\%$**. Es el doble de consistente.
* Además, el Random Forest tuvo mayor **Precision (80.47% vs 78.74%)**: cuando el modelo dice que alguien sobrevive, se equivoca menos.

Elegir un modelo no es buscar el número más inflado en tu computadora, sino el que tenga **menor probabilidad de fallar cuando se enfrente a datos nunca vistos**.

---

## 5. Honestidad Intelectual: La trampa del 100% en Kaggle

En la competencia de Titanic de Kaggle, si mirás la parte más alta de la tabla de posiciones, vas a ver cientos de personas con un puntaje de **1.00000 (100% de exactitud)**.
¿Descubrieron la inteligencia artificial definitiva? **No.** Como el Titanic ocurrió en 1912, la lista real de supervivientes es de dominio público en Wikipedia. La gente tramposa busca los nombres de `test.csv` en Google, pega las respuestas a mano y sube el archivo para figurar primeros.

**Nuestra meta:** Obtener un **~80-83% honesto**. En el mundo real, los problemas complejos de negocio tienen incertidumbre irreductible y ruido. Saber aceptar los límites de los datos y reportar con transparencia métricas de validación cruzada reales es lo que diferencia a un profesional de un aficionado.
