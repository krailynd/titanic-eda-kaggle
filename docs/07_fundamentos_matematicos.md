# 📐 Fundamentos Matemáticos y Estadísticos del Pipeline

Este documento detalla todas las **matemáticas, estadísticas y algoritmos de optimización** aplicados en el proyecto **Titanic EDA Kaggle**.

Para cada concepto se presentan dos perspectivas:
1. **La intuición práctica (para quien no tiene formación matemática avanzada):** analogías y explicaciones en lenguaje natural.
2. **El rigor formal (para quien domina la matemática):** fórmulas, deducciones analíticas y su vínculo exacto con el código en `src/`.

---

## 🗺️ Mapa de las Matemáticas Utilizadas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MATEMÁTICAS EN TITANIC EDA ML                         │
├───────────────────────┬──────────────────────────────┬──────────────────────┤
│ 1. Estadística        │ 2. Álgebra Lineal &         │ 3. Teoría de la      │
│    Descriptiva        │    Cálculo Multivariable     │    Información       │
│    - Mediana vs Media │    - Proyecciones vectoriales│    - Impureza de Gini│
│    - Cuantiles / IQR  │    - Función Sigmoide        │    - Reducción de    │
│    - Estimador Moda   │    - Descenso de Gradiente   │      varianza en     │
│                       │    - Entropía Cruzada        │      ensambles       │
├───────────────────────┼──────────────────────────────┴──────────────────────┤
│ 4. Teoría de la       │ 5. Probabilidad Condicional y Métricas               │
│    Probabilidad       │    - Matriz de Confusión                             │
│    - Odds & Log-Odds  │    - Precision, Recall & Media Armónica (F1)         │
│    - Muestreo         │    - Integral de Curva ROC (AUC)                     │
│      Bootstrap        │                                                      │
└───────────────────────┴──────────────────────────────────────────────────────┘
```

---

## 1. Estadística Descriptiva e Imputación de Datos

### A. La Mediana vs La Media (Tratamiento de `Age` y `Fare`)

#### 💡 Para quien no sabe matemática:
Si en un bar hay 5 personas que ganan \$1,000 al mes y entra Elon Musk (que gana \$100,000,000), el *salario promedio* da \$16,000,000. Pero nadie en ese bar gana 16 millones: la media fue destruida por un valor extremo (*outlier*). La **mediana**, en cambio, ordena a todos de menor a mayor y toma el número del medio (\$1,000), ignorando los extremos absurdos.

En el Titanic, el boleto promedio costaba £32, pero había pasajes de £512 y otros gratis (£0). La edad promedio de los niños con título `Master` era 4.5 años. Usar la media hubiera inflado las edades.

#### 🔬 Rigor Formal:
Sean $X = \{x_1, x_2, \dots, x_n\}$ variables aleatorias continuas ordenadas de forma no decreciente $x_{(1)} \le x_{(2)} \le \dots \le x_{(n)}$.

* **Media Aritmética ($\bar{x}$):**
  $$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$
  *Propiedad:* Minimiza la pérdida cuadrática $L_2$: $\arg\min_c \sum_{i=1}^n (x_i - c)^2$. Es altamente sensible a colas pesadas (asimetría o *skewness* positiva).

* **Mediana ($Me$):**
  $$Me(X) = \begin{cases} 
  x_{\left(\frac{n+1}{2}\right)} & \text{si } n \text{ es impar} \\ 
  \frac{1}{2}\left(x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}\right) & \text{si } n \text{ es par} 
  \end{cases}$$
  *Propiedad:* Minimiza la pérdida de desviación absoluta $L_1$: $\arg\min_c \sum_{i=1}^n |x_i - c|$. Es un estimador robusto con un punto de ruptura (*breakdown point*) del **50%**.

#### 💻 En el código (`src/features.py`):
```python
# Imputación condicional de edad por grupo (Pclass, Title) usando la mediana
age_medians = train.groupby(["Pclass", "Title"])["Age"].median()
train["Age"] = train.apply(
    lambda r: age_medians.get((r["Pclass"], r["Title"]), global_median) 
    if pd.isna(r["Age"]) else r["Age"], axis=1
)
```

---

### B. Prevención Matemática del Data Leakage

#### 🔬 Rigor Formal:
Sea $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^{n}$ el conjunto de entrenamiento y $\mathcal{D}_{\text{test}} = \{(x_j^*)\}_{j=1}^{m}$ el conjunto de prueba.
Cualquier función de imputación o transformación $\mathcal{T}_{\theta}(X)$ parametrizada por $\theta$ debe calcularse bajo la condición:
$$\hat{\theta} = f(\mathcal{D}_{\text{train}})$$
$$\mathcal{T}_{\hat{\theta}}(\mathcal{D}_{\text{test}}) = g(\mathcal{D}_{\text{test}}; \hat{\theta})$$
Si $\theta$ se calcula sobre $\mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{test}}$, la distribución empírica $\hat{P}_{\text{train}}$ se contamina con información de soporte de $\hat{P}_{\text{test}}$, violando el supuesto de independencia y arrojando una estimación sesgada del error de generalización $\mathbb{E}[L(Y, \hat{f}(X))]$.

---

## 2. Álgebra Lineal: Estandarización y One-Hot Encoding

### A. Estandarización Z-Score (Normalización Gaussiana)

#### 💡 Para quien no sabe matemática:
La edad va de 0 a 80 años, pero la tarifa va de 0 a 512 libras. Si le das esos números a la Regresión Logística, el modelo va a pensar que la tarifa es "5 veces más importante" solo porque el número es más grande. La estandarización convierte todas las variables a una escala común con promedio 0 y dispersión 1, como si pusieras todas las monedas en la misma divisa.

#### 🔬 Rigor Formal:
Para cada variable continua $j \in \{\text{Age}, \text{Fare}, \text{FamilySize}\}$:
$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
Donde la esperanza muestral $\mu_j$ y la desviación estándar $\sigma_j$ son:
$$\mu_j = \frac{1}{n}\sum_{i=1}^n x_{ij}, \quad \sigma_j = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_{ij} - \mu_j)^2}$$

*Geometría de la Optimización:*  
En la Regresión Logística, las curvas de nivel de la función de pérdida sin estandarizar forman elipsoides muy alargados. Esto produce oscilaciones perjudiciales en el gradiente descendente. Al estandarizar ($Z = (X-\mu)/\sigma$), la matriz Hessiana $H = \nabla^2 \mathcal{L}$ pasa a tener un número de condición $\kappa(H) \approx 1$, haciendo que las curvas de nivel sean hiperesferas casi perfectas y el gradiente apunte directo al mínimo global en menos iteraciones.

#### 💻 En el código (`src/model.py`):
```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # z = (x - mu) / sigma
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
```

---

### B. One-Hot Encoding como Base Canónica en $\mathbb{R}^k$

Para variables cualitativas con $k$ categorías ($C_1, \dots, C_k$), se proyecta cada categoría a la base ortonormal estándar del espacio vectorial euclidiano:
$$\phi(C_j) = \mathbf{e}_j = [0, \dots, 1, \dots, 0]^T \in \mathbb{R}^k$$
Para evitar la multicolinealidad perfecta (la trampa de la variable ficticia o *dummy variable trap*, donde $\sum_{j=1}^k \mathbf{e}_j = \mathbf{1}$), se descarta una categoría base (`drop_first=True`), proyectando el espacio a $\mathbb{R}^{k-1}$.

---

## 3. Modelo 1: Regresión Logística (Probabilidad y Optimización)

La Regresión Logística modela la probabilidad a posteriori de que un pasajero sobreviva ($Y=1$) dado su vector de características $x \in \mathbb{R}^{22}$.

### A. La Función Sigmoide y los Log-Odds

#### 🔬 Rigor Formal:
Partimos del concepto de **Odds** (posibilidad relativa):
$$\operatorname{Odds}(p) = \frac{p}{1-p}$$
Aplicando el logaritmo natural obtenemos el **Logit** (Log-Odds):
$$\operatorname{Logit}(p) = \ln\left(\frac{p}{1-p}\right) = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{22} w_j x_j + b$$
Despejando la probabilidad $p = P(Y=1|\mathbf{x})$:
$$P(Y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$

```
Probabilidad p
  1.0 ┤                                  ╭─────────── (Sobrevive: p > 0.5)
      │                                ╭─╯
  0.5 ┼───────────────────────────────╭╯───────────── (Umbral de decisión z=0)
      │                            ╭──╯
  0.0 ┴────────────╭───────────────╯───────────────── (Fallece: p < 0.5)
      └────────────┴───────────────┴───────────────┴─
                  z = -∞          z = 0           z = +∞
                        Combinación lineal w^T x + b
```

---

### B. Función de Costo: Entropía Cruzada Binaria con Regularización $L_2$

Asumiendo que las etiquetas $y_i \in \{0, 1\}$ siguen una distribución Bernoulli independiente condicionada a $x_i$, la verosimilitud conjunta es:
$$L(\mathbf{w}, b) = \prod_{i=1}^n \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}, \quad \text{donde } \hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b)$$

Tomando el logaritmo negativo ($-\ln L$) y agregando la penalización $L_2$ (Ridge) con coeficiente $\lambda = 1/C$:
$$\mathcal{J}(\mathbf{w}, b) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

#### ⚙️ Derivada y Gradiente:
El gradiente respecto a cada peso $w_j$ es notablemente elegante:
$$\frac{\partial \mathcal{J}}{\partial w_j} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i) x_{ij} + \lambda w_j$$
*Interpretación:* La corrección del peso es proporcional al **error de predicción** $(\hat{y}_i - y_i)$ multiplicado por el valor de la característica $x_{ij}$.

---

## 4. Modelo 2: Random Forest (Teoría de la Información y Ensamble)

### A. Impureza de Gini en los Árboles de Decisión

#### 💡 Para quien no sabe matemática:
Imaginate que tenés una bolsa con canicas rojas (fallecidos) y verdes (sobrevivientes).
* Si todas las canicas son verdes, la bolsa es **100% pura** (Gini = 0). No hay incertidumbre.
* Si la mitad son verdes y la mitad rojas, la bolsa tiene **máxima confusión** (Gini = 0.5).
El algoritmo busca cortar los datos con la pregunta que deje las bolsas lo más puras posible.

#### 🔬 Rigor Formal:
En un nodo $t$, con $K=2$ clases y probabilidad empírica $p_k = P(Y=k|t)$:
$$I_G(t) = 1 - \sum_{k=1}^K p_k^2 = 1 - (p_0^2 + p_1^2) = 2 p_0 (1 - p_0)$$
El algoritmo evalúa cada variable continua o discreta $j$ y cada posible umbral $\theta$, calculando la **Ganancia de Impureza ($\Delta I_G$)**:
$$\Delta I_G(t, j, \theta) = I_G(t) - \left( \frac{n_L}{n} I_G(t_L) + \frac{n_R}{n} I_G(t_R) \right)$$
Elige la partición óptima:
$$(j^*, \theta^*) = \arg\max_{j, \theta} \Delta I_G(t, j, \theta)$$

---

### B. Reducción Matemática de la Varianza (Bootstrap Aggregating)

¿Por qué Random Forest tuvo una dispersión de **$\pm 1.00\%$** frente al **$\pm 2.08\%$** de la Regresión Logística?

#### 🔬 Demostración Teórica:
Sea un ensamble de $B$ árboles de decisión individuales $T_1(x), T_2(x), \dots, T_B(x)$, cada uno con varianza individual $\sigma^2$ y una correlación promedio entre pares de árboles de $\rho = \operatorname{Corr}(T_i, T_j)$.

La varianza de la predicción promedio $\bar{T}(x) = \frac{1}{B} \sum_{b=1}^B T_b(x)$ es:
$$\operatorname{Var}(\bar{T}(x)) = \operatorname{Var}\left(\frac{1}{B}\sum_{b=1}^B T_b\right) = \frac{1}{B^2} \sum_{b=1}^B \operatorname{Var}(T_b) + \frac{1}{B^2} \sum_{b \ne j} \operatorname{Cov}(T_b, T_j)$$
$$\operatorname{Var}(\bar{T}(x)) = \frac{B \sigma^2}{B^2} + \frac{B(B-1)\rho \sigma^2}{B^2} = \frac{\sigma^2}{B} + \frac{B-1}{B} \rho \sigma^2$$
Reagrupando:
$$\operatorname{Var}(\bar{T}(x)) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

#### 🎯 Conclusiones del Teorema:
1. Al aumentar el número de árboles $B \rightarrow \infty$, el segundo término $\frac{1-\rho}{B}\sigma^2 \rightarrow 0$.
2. La varianza mínima irreducible queda determinada exclusivamente por $\rho \sigma^2$.
3. Para reducir $\rho$ (la correlación entre árboles), Random Forest aplica dos técnicas de aleatorización:
   * **Bagging:** Cada árbol se entrena sobre una muestra aleatoria con reemplazo ($63.2\%$ de datos únicos y $36.8\%$ fuera de bolsa o *Out-of-Bag*).
   * **Subespacio aleatorio de variables:** En cada división de nodo, solo evalúa un subconjunto aleatorio de $\approx \sqrt{d}$ variables.

Esto reduce $\rho$ drásticamente y colapsa la varianza del ensamble a la mitad ($\pm 1.00\%$).

---

## 5. Métricas de Evaluación y Probabilidad Condicional

A partir de la matriz de confusión binaria:
* **$TP$ (Verdaderos Positivos):** Sobrevivió y el modelo predijo que sobrevivía.
* **$FP$ (Falsos Positivos):** Murió pero el modelo predijo que sobrevivía (Error Tipo I).
* **$TN$ (Verdaderos Negativos):** Murió y el modelo predijo que moría.
* **$FN$ (Falsos Negativos):** Sobrevivió pero el modelo predijo que moría (Error Tipo II).

### Fórmulas Matemáticas:

| Métrica | Definición Probabilística | Fórmula Algebraica | Propósito en el Proyecto |
|---|---|---|---|
| **Accuracy** | $P(\hat{Y} = Y)$ | $\frac{TP + TN}{TP + TN + FP + FN}$ | Proporción global de aciertos (82.83%). |
| **Precision** | $P(Y=1 \mid \hat{Y}=1)$ | $\frac{TP}{TP + FP}$ | Certeza al predecir que alguien sobrevive (80.47%). |
| **Recall** | $P(\hat{Y}=1 \mid Y=1)$ | $\frac{TP}{TP + FN}$ | Capacidad de rescatar a los supervivientes reales (73.07%). |
| **F1-Score** | Media armónica de $P$ y $R$ | $\frac{2 \cdot P \cdot R}{P + R} = \frac{2TP}{2TP + FP + FN}$ | Balance ante clases asimétricas (76.49%). |

### ¿Por qué Media Armónica para el F1-Score?
La media aritmética simple entre 100% de Precision y 0% de Recall daría un 50% tramposo.
La media armónica:
$$H(x, y) = \frac{2}{\frac{1}{x} + \frac{1}{y}} = \frac{2xy}{x+y}$$
Se aproxima al menor de los dos valores si uno de ellos se desploma hacia cero. Exige que el modelo sea bueno en ambos frentes a la vez.

---

### Integral del Área bajo la Curva ROC (ROC-AUC = 87.43%)

La curva ROC grafica la tasa de verdaderos positivos $\operatorname{TPR}(\tau)$ contra la tasa de falsos positivos $\operatorname{FPR}(\tau)$ para todos los umbrales de decisión posibles $\tau \in [0, 1]$.

#### 🔬 Teorema Probabilístico del AUC:
El área bajo la curva ROC equivale a la probabilidad de que el modelo asigne un puntaje de riesgo más alto a un sobreviviente tomado al azar que a un fallecido tomado al azar:
$$\operatorname{AUC} = \int_0^1 \operatorname{TPR}(\operatorname{FPR}^{-1}(u)) \, du = P\left(f(X^+) > f(X^-)\right)$$
Un **AUC de 87.43%** significa que si tomás un superviviente real y un fallecido al azar, en el **87.4% de los casos** el modelo le asignará mayor probabilidad de sobrevivir al que realmente sobrevivió.
