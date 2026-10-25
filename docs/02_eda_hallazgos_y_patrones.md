# 🔍 Análisis Exploratorio de Datos (EDA) — Hallazgos y Patrones

Este documento sintetiza la evidencia empírica descubierta durante la fase de exploración del dataset del Titanic (891 pasajeros en `train.csv`).

---

## 1. La Tasa Global de Supervivencia

De los 891 pasajeros registrados en el conjunto de entrenamiento:
* **Fallecieron:** 549 pasajeros (**61.62%**).
* **Sobrevivieron:** 342 pasajeros (**38.38%**).

Esta asimetría basal implica que cualquier modelo ingenuo que prediga "nadie sobrevive" obtendría un 61.6% de exactitud aparente, demostrando la necesidad de evaluar métricas equilibradas como Precision, Recall, F1 y ROC-AUC.

---

## 2. El Protocolo Marítimo: "Mujeres y Niños Primero"

La variable `Sex` es, por amplio margen, el predictor individual más determinante del destino de los pasajeros:

| Sexo | Pasajeros | Sobrevivientes | Tasa de Supervivencia |
|---|---|---|---|
| **Femenino (female)** | 314 | 233 | **74.20%** |
| **Masculino (male)** | 577 | 109 | **18.89%** |

* Casi 3 de cada 4 mujeres sobrevivieron.
* Menos de 1 de cada 5 hombres sobrevivió.
* La tripulación aplicó de forma estricta la norma de evacuación hacia los botes salvavidas.

---

## 3. Disparidad Socioeconómica: Supervivencia por Clase (`Pclass`)

La ubicación física y el estatus social determinaron tanto el acceso temprano a la cubierta superior como la disponibilidad de botes salvavidas:

| Clase | Pasajeros | Sobrevivientes | Tasa de Supervivencia | Tarifa Promedio (£) |
|---|---|---|---|---|
| **1ra Clase (Upper)** | 216 | 136 | **62.96%** | £84.15 |
| **2da Clase (Middle)** | 184 | 87 | **47.28%** | £20.66 |
| **3ra Clase (Lower)** | 491 | 119 | **24.24%** | £13.68 |

* Un pasajero de **Primera Clase** tuvo más de **2.6 veces** más probabilidades de salvarse que uno de **Tercera Clase**.
* En Tercera Clase viajaba la mayor cantidad de personas (más del 55% del pasaje), ubicadas en los compartimentos más profundos del casco, dificultando la evacuación nocturna.

---

## 4. Título Social Extraído del Nombre (`Title`)

Al parsear el nombre de cada persona, los títulos revelaron información simultánea de género, edad y estatus nobiliario o profesional:

| Título | Significado / Perfil | Tasa de Supervivencia |
|---|---|---|
| **Mrs** | Mujeres casadas / viudas | **79.20%** |
| **Miss** | Mujeres jóvenes / solteras | **69.78%** |
| **Master** | Niños varones (menores de 14 años) | **57.50%** |
| **Mr** | Hombres adultos | **15.67%** |
| **Rare** | Nobles, militares, clérigos y médicos | **34.78%** |

**Hallazgo crucial:** Mientras que los varones adultos (`Mr`) tenían apenas un 15.67% de supervivencia, los niños varones identificados con el título `Master` alcanzaron un **57.50%**. Esto demuestra que la variable `Title` encapsula información combinada de sexo y edad con mayor poder predictivo que la variable `Age` cruda.

---

## 5. Dinámica Familiar (`FamilySize` e `IsAlone`)

Construyendo la variable derivada `FamilySize = SibSp + Parch + 1`:

```
Tasa de Supervivencia según Tamaño de Familia:
1 persona (Solo):        30.35% (537 pasajeros)
2 personas:              55.28% (161 pasajeros)  <--- Zona Óptima
3 personas:              57.84% (102 pasajeros)  <--- Zona Óptima
4 personas:              72.41% ( 29 pasajeros)  <--- Zona Óptima
5 personas:              20.00% ( 15 pasajeros)  <--- Caída Abrupta
6 personas:              13.64% ( 22 pasajeros)
7 personas:              33.33% ( 12 pasajeros)
8 personas:               0.00% (  6 pasajeros)
11 personas:              0.00% (  7 pasajeros)
```

### Interpretación sociológica:
* **Viajar solo (`IsAlone = 1`):** Penaliza la supervivencia (~30%). Muchas personas solas eran hombres jóvenes de tercera clase.
* **Familias nucleares pequeñas (2 a 4 integrantes):** Tuvieron la mayor probabilidad de éxito (>55% a 72%). Los acompañantes pudieron asistirse mutuamente y ganar acceso prioritario a los botes.
* **Familias numerosas ($\ge 5$ integrantes):** La tasa se desploma violentamente por debajo del 20%. Mantener unido a un grupo grande en la oscuridad e incertidumbre impidió alcanzar los botes a tiempo.

---

## 6. Puerto de Embarque (`Embarked`) y Cubiertas (`Cabin`)

* **Cherbourg (`C`):** Mostró la mayor tasa de supervivencia (**55.36%**). Esto se debe a que una proporción desproporcionada de pasajeros de primera clase embarcó en Francia.
* **Queenstown (`Q`):** Tasa de **38.96%**, compuesto mayoritariamente por emigrantes irlandeses en 3ra clase.
* **Southampton (`S`):** Tasa de **33.70%**, puerto principal donde abordó el grueso de la tripulación y pasajeros de clase trabajadora.
* **Cabinas (`Cabin`):** El 77% de los datos es nulo. Sin embargo, el solo hecho de **poseer un camarote registrado** (cubiertas A, B, C, D, E) correlaciona fuertemente con la primera clase y una supervivencia superior al 66%, frente al ~30% de los pasajeros con cabina desconocida (`Deck = 'U'`).
