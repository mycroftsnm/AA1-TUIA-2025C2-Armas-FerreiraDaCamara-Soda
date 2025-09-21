# ============================================================================
# PROYECTO: PREDICCIÓN DE TARIFAS UBER CON REGRESIÓN LINEAL MÚLTIPLE
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARGA Y EXPLORACIÓN INICIAL DEL DATASET
# ============================================================================
print("="*60)
print("1. CARGA Y EXPLORACIÓN INICIAL")
print("="*60)

# Cargar datos
df = pd.read_csv('uber_fares.csv')
print(f"Shape del dataset: {df.shape}")
print("\nPrimeras 5 filas:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())
print("\nValores nulos por columna:")
print(df.isnull().sum())
print("\nEliminado de valores faltantes:")
df.dropna(inplace=True)
print(f"Nuevo shape del dataset: {df.shape}")
print("\nEstadísticas descriptivas:")
print(df.describe())

# TODO: Agregar análisis exploratorio más detallado
# - Verificar valores nulos
# - Detectar outliers
# - Visualizaciones iniciales

# ============================================================================
# 2. PROCESAMIENTO Y FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*60)
print("2. PROCESAMIENTO Y FEATURE ENGINEERING")
print("="*60)

# 2.1 Procesamiento de fechas (usando nuestro código anterior)
df['date'] = pd.to_datetime(df['pickup_datetime']).dt.floor('s')

# Variables temporales básicas
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Codificación cíclica
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

# Variables de contexto
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Análisis de demanda por hora
hourly_trip_counts = df.groupby('hour').size().reset_index(name='trips_per_hour')
df = df.merge(hourly_trip_counts, on='hour', how='left')

q33 = hourly_trip_counts['trips_per_hour'].quantile(0.33)
q66 = hourly_trip_counts['trips_per_hour'].quantile(0.66)

def classify_demand(trips):
    if trips <= q33:
        return 'low'
    elif trips <= q66:
        return 'medium'
    else:
        return 'high'

df['demand_level'] = df['trips_per_hour'].apply(classify_demand)
demand_dummies = pd.get_dummies(df['demand_level'], prefix='demand')
df = pd.concat([df, demand_dummies], axis=1)

df['is_peak_hour'] = df['trips_per_hour'] > q66

# 2.2 Cálculo de distancia (Fórmula de Haversine)
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia entre dos puntos en la Tierra usando la fórmula de Haversine
    """
    R = 6371  # Radio de la Tierra en km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    
    return distance

df['distance'] = haversine_distance(
    df['pickup_latitude'], df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)

# 2.3 Selección de features finales
final_features = [
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude',
    'distance', 'passenger_count',
    'hour_sin', 'hour_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'month_sin', 'month_cos',
    'is_weekend', 'is_peak_hour',
    'demand_high', 'demand_medium'
]

print(f"Features seleccionadas: {len(final_features)}")
print(final_features)

# ============================================================================
# 3. DIVISIÓN DEL DATASET Y ESCALADO
# ============================================================================
print("\n" + "="*60)
print("3. DIVISIÓN DEL DATASET Y ESCALADO")
print("="*60)

# Preparar X y y
X = df[final_features].copy()
y = df['fare_amount'].copy()

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado
variables_to_scale = [
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude', 
    'distance', 'passenger_count'
]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[variables_to_scale] = scaler.fit_transform(X_train[variables_to_scale])
X_test_scaled[variables_to_scale] = scaler.transform(X_test[variables_to_scale])

print(f"Train shape: {X_train_scaled.shape}")
print(f"Test shape: {X_test_scaled.shape}")

# ============================================================================
# 4. IMPLEMENTACIÓN DE MODELOS DE REGRESIÓN
# ============================================================================
print("\n" + "="*60)
print("4. MODELOS DE REGRESIÓN")
print("="*60)

# Diccionario para almacenar resultados
results = {}

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Evalúa un modelo y retorna métricas
    """
    # Entrenar
    model.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }
    
    return metrics

# 4.1 REGRESIÓN LINEAL MÚLTIPLE CLÁSICA
print("\n4.1 Regresión Lineal Múltiple")
print("-" * 40)

lr_model = LinearRegression()
results['linear_regression'] = evaluate_model(
    lr_model, X_train_scaled, X_test_scaled, y_train, y_test, 'Linear Regression'
)

print(f"R² Train: {results['linear_regression']['train_r2']:.4f}")
print(f"R² Test:  {results['linear_regression']['test_r2']:.4f}")
print(f"RMSE Test: {results['linear_regression']['test_rmse']:.4f}")

# 4.2 GRADIENTE DESCENDIENTE
print("\n4.2 Métodos de Gradiente Descendiente")
print("-" * 40)

# TODO: Implementar y comparar:
# - Gradiente Descendiente (SGDRegressor con batch completo)
# - Gradiente Descendiente Estocástico 
# - Gradiente Descendiente Mini-batch

# Ejemplo para SGD:
sgd_models = {
    'sgd_full_batch': SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000, random_state=42),
    'sgd_stochastic': SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000, random_state=42),
    'sgd_mini_batch': SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000, random_state=42)
}

# TODO: Implementar curvas de loss vs epochs

# 4.3 MÉTODOS DE REGULARIZACIÓN
print("\n4.3 Métodos de Regularización")
print("-" * 40)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
results['ridge'] = evaluate_model(
    ridge_model, X_train_scaled, X_test_scaled, y_train, y_test, 'Ridge'
)

# Lasso Regression
lasso_model = Lasso(alpha=1.0)
results['lasso'] = evaluate_model(
    lasso_model, X_train_scaled, X_test_scaled, y_train, y_test, 'Lasso'
)

# Elastic Net
elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
results['elastic_net'] = evaluate_model(
    elastic_model, X_train_scaled, X_test_scaled, y_train, y_test, 'Elastic Net'
)

print(f"Ridge R² Test:  {results['ridge']['test_r2']:.4f}")
print(f"Lasso R² Test:  {results['lasso']['test_r2']:.4f}")
print(f"Elastic R² Test: {results['elastic_net']['test_r2']:.4f}")

# ============================================================================
# 5. OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================================================
print("\n" + "="*60)
print("5. OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("="*60)

# TODO: Implementar GridSearchCV para:
# - Ridge: alpha = [0.1, 1.0, 10.0, 100.0, 1000.0]
# - Lasso: alpha = [0.1, 1.0, 10.0, 100.0, 1000.0]
# - SGD: learning_rate, eta0, alpha

# Ejemplo para Ridge:
ridge_param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_param_grid, cv=5, scoring='r2')
ridge_grid.fit(X_train_scaled, y_train)

print(f"Mejor alpha para Ridge: {ridge_grid.best_params_['alpha']}")
print(f"Mejor score para Ridge: {ridge_grid.best_score_:.4f}")

# ============================================================================
# 6. VISUALIZACIONES Y ANÁLISIS DE RESIDUOS
# ============================================================================
print("\n" + "="*60)
print("6. VISUALIZACIONES Y ANÁLISIS")
print("="*60)

def plot_residuals(y_true, y_pred, title):
    """
    Gráfico de residuos
    """
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residuos vs Predicciones
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Predicciones')
    ax1.set_ylabel('Residuos')
    ax1.set_title(f'{title} - Residuos vs Predicciones')
    
    # Q-Q plot de residuos
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title(f'{title} - Q-Q Plot de Residuos')
    
    plt.tight_layout()
    plt.show()

# TODO: Crear gráficos para todos los modelos
# plot_residuals(y_test, results['linear_regression']['y_test_pred'], 'Linear Regression')

# ============================================================================
# 7. COMPARACIÓN DE MODELOS
# ============================================================================
print("\n" + "="*60)
print("7. COMPARACIÓN DE MODELOS")
print("="*60)

# Crear tabla comparativa
comparison_df = pd.DataFrame({
    model_name: {
        'R² Train': metrics['train_r2'],
        'R² Test': metrics['test_r2'], 
        'RMSE Train': metrics['train_rmse'],
        'RMSE Test': metrics['test_rmse'],
        'MAE Train': metrics['train_mae'],
        'MAE Test': metrics['test_mae']
    }
    for model_name, metrics in results.items()
}).T

print(comparison_df.round(4))

# TODO: Seleccionar el mejor modelo basado en métricas de test
best_model_name = comparison_df['R² Test'].idxmax()
print(f"\nMejor modelo: {best_model_name}")

# ============================================================================
# 8. CONCLUSIONES
# ============================================================================
print("\n" + "="*60)
print("8. CONCLUSIONES")
print("="*60)

print("""
TODO: Completar con análisis de resultados:

8.1 Análisis de Gradiente Descendiente:
- ¿Cuál converge más rápido?
- ¿Diferencias en performance final?
- Análisis de curvas loss vs epochs

8.2 Análisis de Regularización:
- ¿Mejora el overfitting?
- ¿Qué features se penalizan más en Lasso?
- ¿Cuál es el mejor balance bias-variance?

8.3 Comparación General:
- Modelo con mejor performance en test
- Trade-offs entre interpretabilidad y performance
- Recomendaciones finales
""")