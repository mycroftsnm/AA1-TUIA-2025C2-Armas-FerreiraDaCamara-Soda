# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: aa1-tuia-2025c2-armas-ferreiradacamara-soda
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve, f1_score)

# %%
# Carga el dataset en un dataframe
df = pd.read_csv('weatherAUS.csv')

# Revisa si hay filas duplicadas
df.duplicated().sum() # 0 filas duplicadas

pd.set_option('display.max_columns', None)
df.describe(include='all')

# %%
df.describe()

# %%
print(f"El dataframe posee {len(df.columns)} variables:\n" + "\n".join(f"  - {col}" for col in df.columns))

# %% [markdown]
# # Limpieza y preprocesamiento

# %% [markdown]
# Análsis de faltantes:

# %%
faltantes_df = pd.DataFrame({
    'NaN': df.isna().sum(),
    '%': (df.isna().sum() / len(df) * 100).round(2)
}).sort_values('NaN', ascending=False)
faltantes_df

# %%
faltantes_df = pd.DataFrame({
    'NaN': df.isna().sum(),
    '%': (df.isna().sum() / len(df) * 100).round(2)
}).sort_values('NaN', ascending=False)
faltantes_df

# %%
plt.figure(figsize=(10, 5))
ax = sns.barplot(x=faltantes_df.index, y=faltantes_df['%'], color='coral')
plt.title('Porcentaje de valores faltantes por variable')
plt.xlabel('Variables')
plt.ylabel('% de NaN')
plt.xticks(rotation=45, ha='right')

for i, idx in enumerate(faltantes_df.index):
    ax.text(i, faltantes_df.loc[idx, '%'], f'{int(faltantes_df.loc[idx, "NaN"])}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Todas las variables poseen algún valor faltante con excepción de `Date` y de `Location`. Observamos que `Sunshine`, `Evaporation`, `Cloud3pm` y `Cloud9am` son las que presentan más valores faltantes: 47.69%, 42.79%, 40.15%, 37.74% respectivamente.

# %%
nan_por_fila = df.isna().sum(axis=1)

distribucion_nan = pd.DataFrame({
    'Cantidad_filas': nan_por_fila.value_counts().sort_index(),
    '%': (nan_por_fila.value_counts(normalize=True) * 100).sort_index().round(2)
})

distribucion_nan

# %%
nan_por_fila = df.isna().sum(axis=1)

distribucion_nan = pd.DataFrame({
    'Cantidad_filas': nan_por_fila.value_counts().sort_index(),
    '%': (nan_por_fila.value_counts(normalize=True) * 100).sort_index().round(2)
})

distribucion_nan

# %% [markdown]
# Existen 56420 observaciones sin faltantes, lo que representa un 38.8% del dataset. El resto posee al menos un dato faltante.

# %%
df.info(verbose=True)

# %% [markdown]
# Sunshine

# %%
# distribución de Sunshine según RainTomorrow
df.groupby('RainTomorrow')['Sunshine'].describe(percentiles=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])

# %%
df["RainTomorrow"].value_counts(normalize=True).round(2)

# %% [markdown]
# El dataset está desbalanceado. 78% clase 0 u 22% clase 1.

# %%
# Drop de filas con NaN en la feature objetivo
df = df.dropna(subset=['RainTomorrow'])

# %%
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0}).astype('Int8')
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0}).astype('Int8')

# %% [markdown]
# Análisis de las variables "Cloud"

# %%
df['Cloud3pm'].value_counts(dropna=False)

# %%
df['Cloud9am'].value_counts(dropna=False)

# %% [markdown]
# Por el rango de valores que asumen las variables **Cloud9am** y **Cloud3pm** asumimos que dichas variables están medidas en octas, que es la unidad de medida empleada para describir la nubosidad observable en un determinado lugar. https://es.wikipedia.org/wiki/Octa

# %% [markdown]
# A continuación se imputan valores faltantes de ciudades con los valores de lugares cercanos o incluso que se encuentran dentro de las mismas locaciones. Por ejemplo, completamos faltantes de Sydney con los de SydneyAirport registrados el mismo día.
#
# Identificamos que esto sucede en 5 locaciones:
# - 'SydneyAirport': 'Sydney',
# - 'MelbourneAirport': 'Melbourne',
# - 'PerthAirport': 'Perth',
# - 'Williamtown': 'Newcastle', (Williamtown es el aeropuerto de Newcastle)
# - 'PearceRAAF': 'Perth, (Base aérea muy cerca de Perth)

# %%
# Diccionario para mapear lugares dentro de ciudades (o cercanos) a sus ciudades principales
location_map = {
    'SydneyAirport': 'Sydney',
    'MelbourneAirport': 'Melbourne',
    'PerthAirport': 'Perth',
    'Williamtown': 'Newcastle', # Williamtown es el aeropuerto de Newcastle
    'PearceRAAF': 'Perth'      # Base aérea muy cerca de Perth
}

# Columnas numéricas
numeric_cols = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
    'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 
    'Cloud3pm', 'Temp9am', 'Temp3pm'
]


# Columna 'Date' a formato datetime
df['Date'] = pd.to_datetime(df['Date'])


# Imputación cruzada de datos faltantes

# Hacemos una copia para no modificar el original mientras iteramos
df_imputed = df.copy()

for airport, city in location_map.items():
    # Filtramos los datos solo para el par ciudad/aeropuerto actual
    city_rows = df['Location'] == city
    airport_rows = df['Location'] == airport

    # Creamos vistas temporales alineadas por fecha para facilitar la imputación
    city_data = df[city_rows].set_index('Date')[numeric_cols]
    airport_data = df[airport_rows].set_index('Date')[numeric_cols]
    
    # Rellenamos faltantes en la ciudad con datos del aeropuerto
    imputed_city_data = city_data.fillna(airport_data)
    
    # Rellenamos faltantes en el aeropuerto con datos de la ciudad
    imputed_airport_data = airport_data.fillna(city_data)
    
    # Actualizamos el DataFrame principal con los datos imputados
    # Usamos .reindex como city_data.index para asegurar el alineamiento correcto
    df_imputed.loc[city_rows, numeric_cols] = imputed_city_data.reindex(city_data.index).values
    df_imputed.loc[airport_rows, numeric_cols] = imputed_airport_data.reindex(airport_data.index).values

df = df_imputed

# # Método para agrupar las variables con sus aeropuertos cercanos
# df['Location'].replace(location_map, inplace=True) # Reemplaza los valores que coinciden con las keys del diccionario por sus values

# # Mergeamos las filas duplicadas (mismo día y locación).
# # Para las numéricas, calculamos la media. Para el resto (categóricas), tomamos el primer valor no nulo.
# agg_functions = {col: 'mean' for col in numeric_cols}
# categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# # No queremos agrupar 'Location' ya que es una de nuestras claves de agrupación
# if 'Location' in categorical_cols:
#     categorical_cols.remove('Location')

# for col in categorical_cols:
#     agg_functions[col] = 'first'

# # Agrupamos por fecha y locación y aplicamos las funciones de agregación
# df_final = df.groupby(['Date', 'Location']).agg(agg_functions).reset_index()

# print(f"Dimensiones finales del DataFrame: {df_final.shape}")

# %%
def generar_csv_coordenadas(df):
    import time
    import pandas as pd
    from geopy.geocoders import Nominatim

    ubicaciones = df['Location'].unique()
    australia_coords = pd.DataFrame({"location": ubicaciones})

    geolocator = Nominatim(user_agent="australia_mapper")

    lats, lons = [], []

    def normalizar_nombre_ubicacion(ubicacion):
        for i in range(1, len(ubicacion)):
            if ubicacion[i].isupper():
                return ubicacion[:i] + " " + ubicacion[i:]
        return ubicacion

    nombres_ubicaciones =  map(normalizar_nombre_ubicacion, ubicaciones)

    for ubicacion in nombres_ubicaciones:
        result = geolocator.geocode(f"{ubicacion}, Australia", timeout=10)
        if result:
            lats.append(result.latitude)
            lons.append(result.longitude)
        else:
            print('No se encontró', ubicacion)
            lats.append(None)
            lons.append(None)
        time.sleep(1.1)  # máx 1 req/s


    australia_coords["lat"] = lats
    australia_coords["lon"] = lons

    australia_coords.to_csv("australian_locations.csv", index=False)

# %%
# generar_csv_coordenadas(df) # Descomentar para generar el CSV

# %%
# Df con coordenadas
australia_coords = pd.read_csv("australian_locations.csv")

# Genera variable frecuencia para cada ubicación
australia_coords['frecuencia'] = df['Location'].value_counts().values

# %%
import plotly.express as px

fig = px.scatter_geo(
    australia_coords,
    lat='lat',
    lon='lon',
    scope='oceania',
    color='frecuencia',
    hover_name='location',
    projection='natural earth',
    color_continuous_scale='Purp',
)

# Ajusta los límites del mapa para centrarse en Australia
fig.update_geos(
    lonaxis=dict(range=[min(australia_coords['lon'])-5, max(australia_coords['lon'])+5]),
    lataxis=dict(range=[min(australia_coords['lat'])-5, max(australia_coords['lat'])+5]),
)
fig.update_layout(width=1600,height=900)

fig.update_traces(marker_size=20)

fig.show()

# %% [markdown]
# Observamos que tenemos datos de muchas ubicaciones distintas, implicando que tendremos que generar una gran cantidad de variables dummys lo que corre riesgo de overfitting. Vamos a reducir la dimensionalidad agrupando ubicaciones según sus tipos de clima, siguiendo la clasificación de Koppen. 

# %%
# Genera una nueva variable Climate basada en la clasificación de Koppen, utilizando la variable Location

location_koppen = {
    'Adelaide': 'Temperate',
    'Albany': 'Temperate',
    'Albury': 'Temperate',
    'AliceSprings': 'Arid',
    'BadgerysCreek': 'Temperate',
    'Ballarat': 'Temperate',
    'Bendigo': 'Temperate',
    'Brisbane': 'Temperate',
    'Cairns': 'Tropical',
    'Canberra': 'Temperate',
    'Cobar': 'Arid',
    'CoffsHarbour': 'Temperate',
    'Dartmoor': 'Temperate',
    'Darwin': 'Tropical',
    'GoldCoast': 'Temperate',
    'Hobart': 'Temperate',
    'Katherine': 'Tropical',
    'Launceston': 'Temperate',
    'Melbourne': 'Temperate',
    'Mildura': 'Arid',
    'Moree': 'Temperate',
    'MountGambier': 'Temperate',
    'MountGinini': 'Temperate',
    'Newcastle': 'Temperate',
    'Nhil': 'Temperate',
    'NorahHead': 'Temperate',
    'NorfolkIsland': 'Temperate',
    'Nuriootpa': 'Temperate',
    'PearceRAAF': 'Temperate',
    'Penrith': 'Temperate',
    'Perth': 'Temperate',
    'PerthAirport': 'Temperate',
    'Portland': 'Temperate',
    'Richmond': 'Temperate',
    'Sale': 'Temperate',
    'SalmonGums': 'Arid',
    'Sydney': 'Temperate',
    'SydneyAirport': 'Temperate',
    'Townsville': 'Tropical',
    'Tuggeranong': 'Temperate',
    'Uluru': 'Arid',
    'WaggaWagga': 'Temperate',
    'Walpole': 'Temperate',
    'Watsonia': 'Temperate',
    'Williamtown': 'Temperate',
    'Witchcliffe': 'Temperate',
    'Wollongong': 'Temperate',
    'Woomera': 'Arid',
}

# %%
# Genera la nueva variable en el df original y en el df de coordenadas
df['Climate'] = df['Location'].map(location_koppen)

australia_coords['Climate'] = australia_coords['location'].map(location_koppen)

# %%
import plotly.express as px

fig = px.scatter_geo(
    australia_coords,
    lat='lat',
    lon='lon',
    scope='oceania',
    color='Climate',
    hover_name='location',
    projection='natural earth',
    size='frecuencia',
)

# Ajusta los límites del mapa para centrarse en Australia
fig.update_geos(
    lonaxis=dict(range=[min(australia_coords['lon'])-5, max(australia_coords['lon'])+5]),
    lataxis=dict(range=[min(australia_coords['lat'])-5, max(australia_coords['lat'])+5]),
)
fig.update_layout(width=1600,height=900)

fig.show()

# %% [markdown]
# ### Split Train/Test

# %%
# Separa el 80% para train y 20% para test
train, test= train_test_split(df, test_size=0.2, random_state=1)

# %% [markdown]
# # EDA

# %%
variables_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print(f"Hay {len(variables_numericas)} variables_numericas:\n{variables_numericas}")

# %%
# Distribución de variables
fig, axes = plt.subplots(4, 4, figsize=(20, 18))

for i, var in enumerate(variables_numericas):
    if var == 'Cloud3pm' or var == 'Cloud9am':
        sns.countplot(data=train, x=var, hue='Climate', palette='muted', ax=axes[i // 4, i % 4], hue_order=['Arid', 'Temperate', 'Tropical'])
    else:
        sns.kdeplot(data=train, x=var, hue='Climate', palette='muted', ax=axes[i // 4, i % 4], hue_order=['Arid', 'Temperate', 'Tropical'], common_norm=False)

plt.tight_layout()
plt.show()

# %%
train[variables_numericas]

# %%
fig, ax1 = plt.subplots(figsize=(16, 9))

sns.heatmap(data=train[variables_numericas].corr(), ax=ax1, annot=True, vmin=-1, vmax=1)

plt.tight_layout()
plt.show()

# %%
ayer_segun_hoy = pd.crosstab(train['RainTomorrow'], train['RainToday'], normalize='index')
hoy_segun_ayer = pd.crosstab(train['RainToday'], train['RainTomorrow'], normalize='index')


fig, axes = plt.subplots(1, 2, figsize=(16, 9))

sns.heatmap(hoy_segun_ayer, annot=True, cmap='Purples', fmt='.3f', cbar=False, ax=axes[0])
sns.heatmap(ayer_segun_hoy, annot=True, cmap='Purples', fmt='.3f', cbar=False, ax=axes[1])

axes[0].set_title('Proporción de días que llovió hoy según si llovió ayer')
axes[0].set_xticks(ticks=[0.5, 1.5], labels=['No', 'Sí'])
axes[0].set_yticks(ticks=[0.5, 1.5], labels=['No', 'Sí'])
axes[0].set_xlabel('¿Llovió hoy?')
axes[0].set_ylabel('¿Llovió ayer?')

axes[1].set_title('Proporción de días que llovió ayer según si llovió hoy')
axes[1].set_xticks(ticks=[0.5, 1.5], labels=['No', 'Sí'])
axes[1].set_yticks(ticks=[0.5, 1.5], labels=['No', 'Sí'])
axes[1].set_xlabel('¿Llovió ayer?')
axes[1].set_ylabel('¿Llovió hoy?')

plt.show()

# %%
train['Rainfall'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99, .999, .9999])

# %%
train = train[train['Rainfall'] < 188]
test = test[test['Rainfall'] < 188]


# %%
# Crea los bins para Rainfall
bins = [float('-inf'), 0, 1, 5, float('inf')]

intervalos = pd.cut(train['Rainfall'], bins=bins, right=True)

train['Rainfall_range'] = intervalos
# Convierte los intervalos a strings para que Seaborn pueda manejarlos
train['Rainfall_range'] = train['Rainfall_range'].astype(str)

# Asegura que los rangos mantengan el orden
train['Rainfall_range'] = pd.Categorical(
    train['Rainfall_range'],
    categories=[str(interval) for interval in intervalos.cat.categories],
    ordered=True
)

frecuencias = train['Rainfall_range'].value_counts(normalize=True).sort_index()

fig, ax1 = plt.subplots(figsize=(16, 9))
sns.histplot(
    data=train,
    x='Rainfall_range',
    hue='RainTomorrow',
    palette='muted',
    multiple='fill',  # Mostrar proporciones dentro de cada bin
    ax=ax1,
)

ax1.set_xlabel('Rango de Lluvia (mm)')
ax1.set_ylabel('Proporción de casos que llovió al día siguiente')
ax1.set_title('Distribución de mm de lluvia registrados y si llovió al día siguiente')

ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

ax1.legend(title='', labels=['Llovió al día siguiente', 'No llovió al dia siguiente'], loc='upper right')

# Segundo eje para la proporción absoluta
ax2 = ax1.twinx()
ax2.plot(frecuencias.index, frecuencias, color=sns.color_palette('muted')[3], marker='o', label='Frecuencia relativa')
ax2.legend(loc='upper left')

# Oculta el eje y secundario; tiene la misma escala que el principal.
ax2.set_axis_off()
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Variable **Evaporation**

# %%
train['Evaporation'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99, .999, .9999])

# %%
train = train[train['Evaporation'] < 71]
test = test[test['Evaporation'] < 71]

# %%
# Crea los bins para Evaporation
bins = [float('-inf'), 2.5, 5, 7.5, float('inf')]

intervalos = pd.cut(train['Evaporation'], bins=bins, right=True)

train['Evaporation_range'] = intervalos
# Convierte los intervalos a strings para que Seaborn pueda manejarlos
train['Evaporation_range'] = train['Evaporation_range'].astype(str)

# Asegura que los rangos mantengan el orden
train['Evaporation_range'] = pd.Categorical(
    train['Evaporation_range'],
    categories=[str(interval) for interval in intervalos.cat.categories],
    ordered=True
)

frecuencias = train['Evaporation_range'].value_counts(normalize=True).sort_index()

fig, ax1 = plt.subplots(figsize=(16, 9))
sns.histplot(
    data=train,
    x='Evaporation_range',
    hue='RainTomorrow',
    palette='muted',
    multiple='fill',  # Mostrar proporciones dentro de cada bin
    ax=ax1,
)

ax1.set_xlabel('Rango de Evaporación (mm)')
ax1.set_ylabel('Proporción de casos que llovió al día siguiente')
ax1.set_title('Distribución de mm de evaporación registrados y si llovió al día siguiente')

ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

ax1.legend(title='', labels=['Llovió al día siguiente', 'No llovió al dia siguiente'], loc='upper right')

# Segundo eje para la proporción absoluta
ax2 = ax1.twinx()
ax2.plot(frecuencias.index, frecuencias, color=sns.color_palette('muted')[3], marker='o', label='Frecuencia relativa')
ax2.legend(loc='upper left')

# Oculta el eje y secundario; tiene la misma escala que el principal.
ax2.set_axis_off()
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.show()



# %% [markdown]
# # Análisis de la Influencia de la Dirección del Viento

# %% [markdown]
# ###  Análisis Gráfico de `WindGustDir`

# %%
# Usaremos el dataframe original 'train' antes del preprocesamiento para facilitar la visualización.
# Asegurémonos de que la columna RainTomorrow (tipo Int8) se trate como categórica para el gráfico.
train_eda = train.copy()
train_eda['RainTomorrow'] = train_eda['RainTomorrow'].astype('category')

# Calcular la proporción de lluvia para cada dirección del viento
wind_rain_proportion = train_eda.groupby('WindGustDir')['RainTomorrow'].value_counts(normalize=True).unstack()

# Ordenar por la proporción de lluvia (1.0)
wind_rain_proportion = wind_rain_proportion.sort_values(by=1.0, ascending=False)

# Graficar
plt.figure(figsize=(14, 8))
sns.barplot(x=wind_rain_proportion.index, y=wind_rain_proportion[1.0], palette='viridis', order=wind_rain_proportion.index)
plt.title('Proporción de Días con Lluvia al Día Siguiente por Dirección de Ráfaga de Viento', fontsize=16)
plt.ylabel('Proporción de Lluvia (RainTomorrow = 1)', fontsize=12)
plt.xlabel('Dirección de la Ráfaga de Viento (WindGustDir)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
# **Interpretación del Gráfico:**
# Direcciones norte o noroeste tienen barras ligeramente más altas que otras (este), evidencia que ladirección del viento está asociada con una mayor probabilidad de lluvia al día siguiente.

# %% [markdown]
# ###  Análisis Estadístico (Prueba Chi-Cuadrado)

# %%
from scipy.stats import chi2_contingency

# Crear una tabla de contingencia
contingency_table = pd.crosstab(train['WindGustDir'], train['RainTomorrow'])

# Realizar la prueba de Chi-Cuadrado
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print("--- Prueba de Chi-Cuadrado para WindGustDir y RainTomorrow ---")
print(f"Estadístico Chi2: {chi2:.4f}")
print(f"P-valor: {p_value}")

# Interpretación del p-valor
alpha = 0.05
if p_value < alpha:
    print("\nConclusión: El p-valor es menor que 0.05. Se rechaza la hipótesis nula.")
    print("Existe una asociación estadísticamente significativa entre la dirección del viento y si lloverá mañana.")
else:
    print("\nConclusión: El p-valor es mayor que 0.05. No se puede rechazar la hipótesis nula.")
    print("No hay evidencia de una asociación estadísticamente significativa entre la dirección del viento y si lloverá mañana.")




# %% [markdown]
# Hipótesis: el viento es relevante en decirnos si llueve mañana siempre en cuando esté en consonancia con la ubicación de la costa de la ciudad. Recordemos que la gran mayoría de locaciones que poseemos en el dataset son costeras o muy cercanas a una. Por lo tanto, si la costa está al oeste, un viento oeste podría implicar más posibilidad de lluvia, y viceversa. El reciente análisis nos dice que los vientos del este son menos relacionados con lluvia pero tendemos a creer que esto sucede porque simplemente el presente dataset posee menos locaciones con costas al este.
#
# Para obtener la información sobre relevancia de la dirección del viento según costa más cercana, vamos a generar una variable relacionada con las costa.

# %%
direccion_costa = {
    # --- Costa Este (E) ---
    'Brisbane': 'E', 'Canberra': 'E', 'CoffsHarbour': 'E', 'GoldCoast': 'E',
    'MountGinini': 'E', 'Newcastle': 'E', 'NorahHead': 'E', 'NorfolkIsland': 'E',
    'Penrith': 'E', 'Richmond': 'E', 'Sydney': 'E', 'SydneyAirport': 'E',
    'Tuggeranong': 'E', 'Williamtown': 'E', 'Wollongong': 'E', 'BadgerysCreek': 'E',

    # --- Costa Oeste (W) ---
    'Perth': 'W', 'PerthAirport': 'W', 'PearceRAAF': 'W', 'Witchcliffe': 'W',

    # --- Costa Norte (N) ---
    'Cairns': 'N', 'Darwin': 'N', 'Katherine': 'N', 'Launceston': 'N', 'Townsville': 'N',

    # --- Costa Sur (S) ---
    'Adelaide': 'S', 'Albany': 'S', 'Dartmoor': 'S', 'Hobart': 'S', 'Melbourne': 'S',
    'MountGambier': 'S', 'Nuriootpa': 'S', 'Portland': 'S', 'Sale': 'S',
    'Walpole': 'S', 'Watsonia': 'S',

    # --- Interior (Inland) ---
    'Albury': 'Inland', 'AliceSprings': 'Inland', 'Ballarat': 'Inland',
    'Bendigo': 'Inland', 'Cobar': 'Inland', 'Mildura': 'Inland',
    'Moree': 'Inland', 'Nhil': 'Inland', 'SalmonGums': 'Inland',
    'Uluru': 'Inland', 'WaggaWaga': 'Inland', 'Woomera': 'Inland'
}

# %% [markdown]
# Convertimos las variables direccionales en componentes de seno y coseno, porque son cíclicas. Con esto obtenemos dos grandes ventajes, primero no generamos 15 variables dummies (1 para cada direccion) y la otra ventaja es que captamos bien la naturaleza cíclica de la dirección del viento.

# %%
# Mapeo de las 16 direcciones del viento a ángulos en grados
wind_dir_map = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

wind_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

# Iteramos sobre ambos dataframes (train y test)
for df in [train, test]:
    for col in wind_cols:
        # Mapear dirección a ángulo
        angles = df[col].map(wind_dir_map)
        
        # Convertir a radianes
        radians = np.deg2rad(angles)
        
        # Calcular seno y coseno (los NaNs se propagarán y los manejaremos después)
        df[f'{col}_sin'] = np.sin(radians)
        df[f'{col}_cos'] = np.cos(radians)
        
print("Variables de seno y coseno creadas.")


# %% [markdown]
# Vamos a generar una variable `IsOnShoreWind` que distinga si el viento viene del mar o de la masa continental.
# Como tenemos tres variables de dirección del viento, vamos a distinguir cual es más importante. Suponemos que `WindGustDir` y `WindDir3pm` son más relevantes que `WindDir9am`

# %%
# Comparamos `WindGustDir`, `WindDir9am` y `WindDir3pm` para ver cuál tiene la relación más fuerte con la lluvia cuando se convierte a una variable onshore/offshore. **Este análisis se realiza solo sobre el conjunto de `train` para evitar data leakage.**

def es_viento_marino(row, wind_col_name):
    coast_dir = row['CoastDirection']
    wind_dir = row[wind_col_name]
    
    if pd.isna(wind_dir) or pd.isna(coast_dir) or coast_dir == 'Inland':
        return 0
    if coast_dir in wind_dir:
        return 1
    else:
        return 0

# Crear una copia del df de entrenamiento para este análisis
train_analysis = train.copy()
train_analysis['CoastDirection'] = train_analysis['Location'].map(direccion_costa)

# Crear las 3 variables candidatas
for col in wind_cols:
    # Imputar NaNs para el análisis (ESTO QUITARLO CUANDO SE HAGA EL IMPUTADO FINAL)
    moda = train_analysis[col].mode()[0]
    train_analysis[col] = train_analysis[col].fillna(moda)
    # Crear la variable onshore/offshore
    train_analysis[f'IsOnshore_{col}'] = train_analysis.apply(es_viento_marino, axis=1, wind_col_name=col)

# --- Comparación Visual y Estadística ---
best_wind_var = ''
max_chi2 = -1

for col in wind_cols:
    onshore_col = f'IsOnshore_{col}'
    
    # Gráfico de Barras
    plt.figure(figsize=(6, 4))
    sns.barplot(data=train_analysis, x=onshore_col, y='RainTomorrow', palette='coolwarm')
    plt.title(f'Probabilidad de Lluvia vs. {onshore_col}')
    plt.ylabel('Proporción de Lluvia')
    plt.xticks([0, 1], ['Offshore', 'Onshore'])
    plt.show()
    
    # Prueba Chi-Cuadrado
    contingency_table = pd.crosstab(train_analysis[onshore_col], train_analysis['RainTomorrow'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"--- Análisis para {onshore_col} ---")
    print(f"Estadístico Chi-Cuadrado: {chi2:.2f}")
    
    if chi2 > max_chi2:
        max_chi2 = chi2
        best_wind_var = col

print(f"\n--- Conclusión del Análisis ---")
print(f"La variable de viento con la asociación más fuerte con la lluvia es: '{best_wind_var}'")
print(f"Usaremos esta variable para crear la característica 'IsOnshoreWind' definitiva.")

# %% [markdown]
# ### Creación Final de las Nuevas Variables en Train y Test
# Ahora que hemos elegido la mejor variable de viento, creamos la característica `IsOnshoreWind` en ambos conjuntos de datos.
#

# %%
for df in [train, test]:
    df['CoastDirection'] = df['Location'].map(direccion_costa)
    df['IsOnshoreWind'] = df.apply(es_viento_marino, axis=1, wind_col_name=best_wind_var)

# Eliminar columnas originales y auxiliares
train.drop(columns=['CoastDirection'] + wind_cols, inplace=True)
test.drop(columns=['CoastDirection'] + wind_cols, inplace=True)

print("Variable 'IsOnshoreWind' creada y columnas originales eliminadas.")

# %% [markdown]
# # Paso 1: Finalización del Preprocesamiento y Feature Engineering

# %%
# Se eliminan columnas que no se usarán para el modelo
# Date: Por ahora porque falta procesarla
# Location: Ya la hemos generalizado con la variable Climate
# Las variables _range que se crearon para el EDA.
train = train.drop(columns=['Date', 'Location', 'Rainfall_range', 'Evaporation_range'])
test = test.drop(columns=['Date', 'Location']) 

# %%
# Separar variables predictoras (X) y objetivo (y)
X_train = train.drop('RainTomorrow', axis=1)
y_train = train['RainTomorrow']
X_test = test.drop('RainTomorrow', axis=1)
y_test = test['RainTomorrow']


# %% [markdown]
# ### 1.1 Función Auxiliar para Imputación de Valores Nulos y poder ver que la regresión logística funcione
# Completa con la mediana para numéricas y la moda para categóricas 

# %%
def imputar_valores_nulos(df_train, df_test):
    """
    Función auxiliar para imputar valores nulos usando la mediana para columnas
    numéricas y la moda para las categóricas.
    Ajusta los imputadores con los datos de entrenamiento y transforma ambos sets.
    """
    print("Iniciando imputación de valores nulos...")
    
    # Identificar tipos de columnas desde el dataframe de entrenamiento
    numerical_cols = df_train.select_dtypes(include=np.number).columns
    categorical_cols = df_train.select_dtypes(include=['object']).columns

    # Crear copias para evitar advertencias de pandas
    train_copy = df_train.copy()
    test_copy = df_test.copy()

    # Imputar variables numéricas
    imputer_numerical = SimpleImputer(strategy='median')
    train_copy[numerical_cols] = imputer_numerical.fit_transform(train_copy[numerical_cols])
    test_copy[numerical_cols] = imputer_numerical.transform(test_copy[numerical_cols])
    print(f"Se imputaron {len(numerical_cols)} columnas numéricas.")

    # Imputar variables categóricas
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    train_copy[categorical_cols] = imputer_categorical.fit_transform(train_copy[categorical_cols])
    test_copy[categorical_cols] = imputer_categorical.transform(test_copy[categorical_cols])
    print(f"Se imputaron {len(categorical_cols)} columnas categóricas.")

    # Verificación final
    print("NaNs restantes en train set después de imputar:", train_copy.isnull().sum().sum())
    print("NaNs restantes en test set después de imputar:", test_copy.isnull().sum().sum())
    
    return train_copy, test_copy

# Aplicar la función de imputación
X_train, X_test = imputar_valores_nulos(X_train, X_test)


# %% [markdown]
# ### 1.2 Codificación de Variables Categóricas (Dummies). ==AUXILIAR TAMBIÉN==

# %%
# Identificar columnas categóricas después de la imputación
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Aplicar One-Hot Encoding
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Alinear las columnas para que test tenga las mismas que train
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0

X_test = X_test[train_cols] # Asegurar el mismo orden de columnas

# %% [markdown]
# ### 1.3 Escalado de Variables Numéricas

# %%
scaler = StandardScaler()

# Se escala todo el dataframe ya que las dummies (0 y 1) no se ven afectadas
# por el escalado estándar de forma que perjudique al modelo.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir de nuevo a DataFrame para mantener los nombres de las columnas
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# %% [markdown]
# # Paso 2: Construcción del Modelo de Regresión Logística

# %%
# Instanciar y entrenar el modelo
log_reg = LogisticRegression(random_state=42, max_iter=1000)#, solver='liblinear') #solver liblinear (descenso por coordenadas), ideal para datasets pequeños o binarios, compatible con regularización L1/L2 
log_reg.fit(X_train, y_train)
# posteriormente probar balanceo con modelo = LogisticRegression(class_weight='balanced')
# balancea asisgnando pesos distintos a la función de costo, no modifica la cantidad de datos en cada clase
# %% [markdown]
# # Paso 3: Evaluación Inicial (Umbral por defecto 0.5)

# %%
# Realizar predicciones
y_pred_class = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1] # Probabilidad de pertenencia a la clase positiva (`Mañana Llueve`)

# %% [markdown]
# ### 3.1 Métricas

# %%
print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("\nAUC-ROC:", roc_auc_score(y_test, y_pred_proba))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred_class))

# %% [markdown]
# ### 3.2 Matriz de Confusión y Curva ROC

# %%
# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No llueve', 'Llueve'],
            yticklabels=['No llueve', 'Llueve'])
plt.ylabel('Valor Real')
plt.xlabel('Predicción')
plt.title('Matriz de Confusión (Umbral 0.5)')
plt.show()

# Curva ROC
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# # Paso 4: Búsqueda y Ajuste del Umbral (Threshold Tuning)

# %%
# Calcular precision, recall para diferentes umbrales
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)

# Convertir a F1-score
# Se ignora el último valor de precision y recall para alinear con thresholds
f1_scores = 2 * recall[:-1] * precision[:-1] / (recall[:-1] + precision[:-1])

# Encontrar el mejor umbral
best_threshold_f1 = thresholds_pr[np.argmax(f1_scores)]
print(f"Mejor umbral para maximizar F1-Score: {best_threshold_f1:.4f}")

# %% [markdown]
# ### 4.1 Visualización del Trade-off

# %%
plt.figure(figsize=(10, 7))
plt.plot(thresholds_pr, precision[:-1], "b--", label="Precision")
plt.plot(thresholds_pr, recall[:-1], "g-", label="Recall")
plt.plot(thresholds_pr, f1_scores, "r-", label="F1-Score", alpha=0.6)
plt.axvline(x=best_threshold_f1, color='purple', linestyle='--', label=f'Mejor Umbral (F1-Score) = {best_threshold_f1:.2f}')
plt.xlabel("Umbral")
plt.title("Precision, Recall y F1-Score vs. Umbral de Decisión")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# %% [markdown]
# # Paso 5: Interpretación de Coeficientes

# %%
# Crear un DataFrame con los coeficientes
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': log_reg.coef_[0]
})

# Calcular Odds Ratios
coefficients['Odds_Ratio'] = np.exp(coefficients['Coefficient'])

# Ordenar por el valor absoluto del coeficiente para ver la importancia
coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values(by='Abs_Coefficient', ascending=False)

# Mostrar los 15 más influyentes
print("Top 15 features más influyentes:")
print(coefficients.head(15).drop('Abs_Coefficient', axis=1))

# %% [markdown]
# # Paso 6: Evaluación Final sobre el Conjunto de Test (con umbral óptimo)

# %%
# Aplicar el umbral óptimo a las probabilidades
y_pred_final = (y_pred_proba >= best_threshold_f1).astype(int)

print(f"Métricas con el umbral óptimo de {best_threshold_f1:.4f}\n")
print("Accuracy:", accuracy_score(y_test, y_pred_final))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred_final))

# %%
# Matriz de Confusión Final
cm_final = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No llueve', 'Llueve'],
            yticklabels=['No llueve', 'Llueve'])
plt.ylabel('Valor Real')
plt.xlabel('Predicción')
plt.title(f'Matriz de Confusión (Umbral {best_threshold_f1:.2f})')
plt.show()



# %%
train.columns
