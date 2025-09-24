
![Logo](https://jcc.dcc.fceia.unr.edu.ar/2023/logo_FCEIA.png)

# AA1-TUIA-2025C2-Armas-FerreiraDaCamara-Soda

## Trabajo Práctico N°1: Modelo Predictivo de Regresión Lineal de Tarifas de Uber

### Facultad de Ciencias Exactas, Ingeniería y Agrimensura
**Tecnicatura en Inteligencia Artificial - Aprendizaje Automático 1**
#### Cuerpo docente:
- Spak, Joel
- Almada, Agustín
- Pellejero, Iván
- Crenna, Giuliano 

---

### Objetivos
El propósito de este trabajo práctico es familiarizarse con la biblioteca **scikit-learn** y sus herramientas para:
- Preprocesamiento de datos.
- Implementación de modelos de regresión lineal con diversos hiperparámetros.
- Evaluación de métricas de regresión.

El objetivo principal es construir un modelo predictivo para estimar las tarifas de viajes de Uber utilizando el dataset proporcionado.

---

### Dataset
El dataset utilizado es `uber_fares.csv`, que contiene información sobre viajes de Uber con las siguientes variables:

#### Características de entrada:
- **key**: Identificador único de cada viaje.
- **pickup_datetime**: Fecha y hora en que se activó el taxímetro.
- **passenger_count**: Número de pasajeros en el vehículo (ingresado por el conductor).
- **pickup_longitude**: Longitud donde se activó el taxímetro.
- **pickup_latitude**: Latitud donde se activó el taxímetro.
- **dropoff_longitude**: Longitud donde se desactivó el taxímetro.
- **dropoff_latitude**: Latitud donde se desactivó el taxímetro.

#### Variable de salida (target):
- **fare_amount**: Costo de cada viaje en USD.

---

### Estructura del Repositorio
El repositorio está organizado de la siguiente manera:
- `/uber_fares.csv`: Dataset original.
- `/TP-regresion-AA1.ipynb`: Notebook de Jupyter con análisis e implementación.
- `README.md`: Este archivo, con la descripción general del trabajo.
---

### Instrucciones de Ejecución
1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/mycroftsnm/AA1-TUIA-2025C2-Armas-FerreiraDaCamara-Soda.git
   ```
2. **Instalar dependencias**:
   Con Python 3.8+ instalado. Instalar las librerías necesarias:
   ```bash
   pip install -r requirements.txt
   ```
   Las dependencias principales incluyen: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `geopy`, `plotly`.

3. **Ejecutar los notebooks**:
   - Abrir los notebooks en Jupyter Notebook o JupyterLab, o IDE de preferencia.
   - Ejecutar los bloques de código en orden: `Limpieza y Preprocesamiento` → `Entrenamiento de Modelos` → `Comparativa de modelos`.

---

### Desarrollo del Trabajo Práctico

#### 1. Análisis Descriptivo
- **Datos faltantes**: Se analizaron y trataron los valores faltantes en el dataset, mediante imputación o eliminación, justificando cada decisión.
- **Generación de variables auxiliares**: Se crearon variables a partir de las originales. Por ejemplo `distance`, en base a las coordenadas de inicio y fin de cada viaje.
- **Datos atípicos**: Se identificaron outliers en variables como `fare_amount`, `passenger_count`, y coordenadas geográficas, utilizando diagramas de caja y criterios estadísticos.
- **Visualizaciones**:
  - Histogramas para la distribución de todas las variables.
  - Scatterplots para explorar relaciones entre variables (e.g., `fare_amount` vs. `distance`).
- **Codificación de variables categóricas**: Se procesó `date` para extraer características relevantes (e.g., horas de más demanda, distribución de la cantidad de viajes por día).
- **Escalado**: Posterior al tratado de outliers, se escalaron las variables numéricas con esacalado estándar para garantizar un mejor rendimiento de los modelos.
- **División de datos**: Se separó el dataset en conjuntos de entrenamiento (80%) y prueba (20%).

#### 2. Implementación de Modelos de Regresión
- **Regresión Lineal Múltiple**:
  - Implementada con `LinearRegression` de scikit-learn.
  - Evaluada con métricas: R², MSE, RMSE, MAE.
- **Métodos de Gradiente Descendente**:
  - Se probaron Descenso por el gradiente clásico, SGD (Stochastic Gradient Descent) y Mini-Batch con diferentes configuraciones.
  - Gráficos de Error vs. Iteraciones generados para analizar la convergencia.
- **Regularización**:
  - Implementados métodos Lasso, Ridge y Elastic Net con un rango amplio de coeficientes de regularización.
  - Gráficos de residuos generados para evaluar el ajuste de los modelos.
  - Selección de hiperparámetros con validación cruzada (Cross-Validation).
- **Análisis**:
  - Se compararon los resultados de entrenamiento y prueba para evaluar sobreajuste/subajuste.
  - Conclusiones sobre el desarrollo de la regresión lineal OLS, el impacto de la regularización y el gradiente descendente.

#### 3. Optimización de Hiperparámetros
- **Gradiente Descendente**: Se variaron tasas de aprendizaje y número de iteraciones.
- **Lasso y Ridge**: Se probaron diferentes valores de alpha para optimizar el equilibrio entre sesgo y varianza.
- Observaciones sobre el impacto de los hiperparámetros en el rendimiento de los modelos.

#### 4. Comparación de Modelos
- Se compararon todos los modelos (LinearRegression, descenso por el gradiente, SGD, Mini-Batch, Lasso, Ridge, ElasticNet) utilizando como métricas principales RMSE y MAE.
- Se justificó la elección del mejor modelo basado en el rendimiento en el conjunto de prueba.

#### 5. Conclusión
- Se redactó una conclusión que resume el rendimiento de los modelos y lo observado con la regularización y la optimización de hiperparámetros, la importancia de un correcto acondicionamiento de los datos previo a la generación de los modelos y elección de variables predictoras. La misma conclusión se encuentra en el final de este README.


### Notas Adicionales
- Todo el código incluye comentarios detallados para explicar cada paso y justificar decisiones.

---

### Conclusión

Para lograr una predicción precisa de las tarifas, se preprocesó el dataset. Se generó la variable **distancia** usando las coordenadas de inicio y fin de cada viaje, siendo esta la más importante en nuestros modelos. Durante el proceso, **eliminamos un 2.25% de los datos** que tenían valores incorrectos en distancia y tarifa. También se imputaron **3133 datos** (0.09% del dataset) que presentaban valores absurdos, NaN o atípicos solo en una de estas variables.

Se observó que la mayoría de los viajes se realizaron en **Nueva York**, por lo que se decidió limitar el dominio del modelo a las coordenadas de esta ciudad para optimizar los resultados. 

Se construyeron modelos con **regresión lineal (OLS)**. En el **descenso por gradiente**, se ajustaron los hiperparámetros hasta alcanzar convergencia y métricas aceptables. En la etapa de **regularización**, el modelo **Ridge** mostró una leve variación en los coeficientes, pero sin cambios significativos en las métricas. Con **Lasso** y **ElasticNet**, no se obtuvieron mejoras (α=0).

OLS es el mejor modelo. Esto puede deberse a la relación lineal fuerte de las variables predictoras y la variables target (principalmente entre distancia y tarifa) y la ausencia de multicolinealidad o variables irrelevantes. 

Todos los modelos lograron explicar cerca del **80% de la variabilidad** de los datos, lo que indica un buen desempeño general. Estos resultados sugieren que el modelo es confiable para predecir tarifas de viajes en Nueva York, aunque podría mejorarse con variables adicionales o técnicas no lineales en futuros análisis.