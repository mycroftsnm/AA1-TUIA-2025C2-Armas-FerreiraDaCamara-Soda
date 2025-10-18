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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
# Carga el dataset en un dataframe
df = pd.read_csv('weatherAUS.csv')

# Revisa si hay filas duplicadas
df.duplicated().sum() # 0 filas duplicadas

pd.set_option('display.max_columns', None)
df.describe(include='all')

# %% [markdown]
# # Limpieza y preprocesamiento

# %%
df.info(verbose=True)

# %%
# Drop de filas con NaN en la feature objetivo
df = df.dropna(subset=['RainTomorrow'])

# %%
# Drop de filas con mas de la mitad de features con valor nulo
df = df[df.isna().sum(axis=1) <= 11]

# %%
df['Cloud3pm'].value_counts(dropna=False)

# %%
df['Cloud9am'].value_counts(dropna=False)

# %% [markdown]
# Por el rango de valores que asumen las variables **Cloud9am** y **Cloud3pm** asumimos que dichas variables están medidas en octas, que es la unidad de medida empleada para describir la nubosidad observable en un determinado lugar. https://es.wikipedia.org/wiki/Octa

# %%
df['Cloud9am'] = df['Cloud9am'].replace(9, np.nan)
df['Cloud3pm'] = df['Cloud3pm'].replace(9, np.nan)


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
# Genera una nueva variable Climate basada en la clásificación de Koppen, utilizando la variable Location

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
    'MelbourneAirport': 'Temperate',
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

sns.set_theme()

for i, var in enumerate(variables_numericas):
    if var == 'Cloud3pm' or var == 'Cloud9am':
        sns.countplot(data=train, x=var, ax=axes[i // 4, i % 4])
    else:
        sns.kdeplot(data=train, x=var, ax=axes[i // 4, i % 4])

fig.suptitle('Distribución de variables numéricas', fontsize=18)

plt.tight_layout()
fig.subplots_adjust(top=0.96) # Espacio vertical para el título
plt.show()

# %% [markdown]
# #### Observaciones iniciales
#
# * Agunas gráficas están fuertemente sesgadas a la derecha, sobretodo ***Rainfall*** y ***Evaporation***.
# * En general las distribuciones muestran signos de multimodalidad, posiblemente debido a datos de distintas estaciones del año o distintos climas.
#
# Vamos a verificar si nuestra clasificación en climas de Koppen explica parte de la multimodalidad.

# %%
fig, axes = plt.subplots(4, 4, figsize=(20, 18))

for i, var in enumerate(variables_numericas):
    if var == 'Cloud3pm' or var == 'Cloud9am':
        sns.countplot(data=train, x=var, hue='Climate', palette='muted', ax=axes[i // 4, i % 4], hue_order=['Arid', 'Temperate', 'Tropical'])
    else:
        sns.kdeplot(data=train, x=var, hue='Climate', palette='muted', ax=axes[i // 4, i % 4], hue_order=['Arid', 'Temperate', 'Tropical'], common_norm=False)

fig.suptitle('Distribución de variables numéricas según tipo de clima', fontsize=18)

plt.tight_layout()
fig.subplots_adjust(top=0.96) # Espacio vertical para el título
plt.show()

# %% [markdown]
# Comprobamos que efectivamente la clasificación de climas según koppen ayudo a disminuir la multimodalidad. #TODO redactar bien

# %% [markdown]
# ## Tratado de Outliers

# %% [markdown]
# #### Variable *Rainfall*

# %%
train['Rainfall'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 0.9999])

# %% [markdown]
# Eliminamos los valores mayores al 99.99% de los datos para su posterior imputación. Además aplicamos transformación logarítmica para reducir el impacto sobre la media y prevenir overfitting en el modelo de regresión logística.

# %%
train['Rainfall'] = np.where(train['Rainfall'] > 101, np.nan, train['Rainfall'])
train['Rainfall_log'] = np.log1p(train['Rainfall'])

test['Rainfall'] = np.where(test['Rainfall'] > 101, np.nan, test['Rainfall'])
test['Rainfall_log'] = np.log1p(test['Rainfall'])

# %% [markdown]
# #### Variable *Evaporation*

# %%
train['Evaporation'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.9999])

# %% [markdown]
# Eliminamos los valores mayores al 99.99% de los datos para su posterior imputación #TODO justificar especifico

# %%
train['Evaporation'] = np.where(train['Evaporation'] > 70, np.nan, train['Evaporation'])
test['Evaporation'] = np.where(test['Evaporation'] > 70, np.nan, test['Evaporation'])

# %% [markdown]
# #### Variable *WindSpeed9am*

# %%
train['WindSpeed9am'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.9999])

# %% [markdown]
# Eliminamos los valores mayores al 99.99% de los datos para su posterior imputación #TODO justificar especifico

# %%
train['WindSpeed9am'] = np.where(train['WindSpeed9am'] > 67, np.nan, train['WindSpeed9am'])

# %% [markdown]
# #### Variable *WindSpeed3pm*

# %%
train['WindSpeed3pm'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.9999])

# %% [markdown]
# #### Variable *WindGustSpeed*

# %%
train['WindGustSpeed'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.9999])

# %%
train[train['WindGustSpeed'] < train['WindSpeed9am']]

# %%
train[train['WindGustSpeed'] < train['WindSpeed3pm']]

# %% [markdown]
# ## Análisis

# %%
fig, ax1 = plt.subplots(figsize=(16, 9))

matriz_correlacion = train[variables_numericas].corr()
mascara = np.triu(np.ones_like(matriz_correlacion, dtype=bool))

sns.heatmap(data=matriz_correlacion, ax=ax1, annot=True, vmin=-1, vmax=1, mask=mascara)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Variable target *RainTomorrow*

# %%
fig, ax = plt.subplots(figsize=(16, 9))

sns.countplot(data=train, x='RainTomorrow', hue='RainTomorrow', stat='percent')

fig.suptitle('Distribución de la variable objetivo RainTomorrow')

plt.tight_layout()
plt.show()

# %% [markdown]
# Tenemos un gran desbalance entre las clases de la variable objetivo, 80/20

# %% [markdown]
# ### Variables *RainToday* y *Rainfall*

# %%
fig, ax = plt.subplots(figsize=(16, 9))

sns.countplot(data=train, x='RainToday', hue='RainToday', stat='percent')

fig.suptitle('Distribución de RainToday')

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

plt.tight_layout()
plt.show()

# %% [markdown]
# > Los nombres de las variables fueron reemplazados de forma que 'Today' representa ayer y 'Tomorrow' hoy para favorecer el entendimiento y la naturalidad de los gráficos.

# %% [markdown]
# En el gráfico de la izquierda observamos la proporción de días en los que llovió o no según si llovió el día anterior. Dicho de otra manera, la probabilidad de que vuelva a llover al día siguiente de un día de lluvia. 
#
# Vemos que la probabilidad de que llueva se triplica pasando de 15,3% a 46,3%. Sin embargo no deja de ser siempre más probable que no llueva a que sí lo haga, sin importar si llovió el día anterior.
#
# En el gráfico de la derecha en cambió tenemos la proporción de días que llovió o no el día anterior dado que llovío o no hoy. En este caso las proporciones dieron muy similares a las del otro gráfico, por lo que el análisis es analogo: Es mas probable que haya llovido ayer si llovió hoy, pero siempre es más probable que no haya llovido ayer.

# %% [markdown]
# Queda claro que el hecho de que haya llovido hoy es importante para predecir si lloverá mañana. Procedemos a analizar la variable *Rainfall* para ver si la cantidad de mm de agua caídos influye en la probabilidad de que llueva mañana.

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
    multiple='fill',  # Mostrar proporciones de RainTomorrow en cada bin
    ax=ax1,
)

ax1.set_xlabel('Rango de Lluvia (mm)')
ax1.set_ylabel('Proporción de casos que llovió al día siguiente')
ax1.set_title('Distribución de mm de lluvia registrados y si llovió al día siguiente')

ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

ax1.legend(title='', labels=['Llovió al día siguiente', 'No llovió al dia siguiente'], loc='upper right')

# Segundo eje para la frecuencia relativa
ax2 = ax1.twinx()
ax2.plot(frecuencias.index, frecuencias, color=sns.color_palette('muted')[3], marker='o', label='Frecuencia relativa')
ax2.legend(loc='upper left')

# Oculta el eje y secundario; tiene la misma escala que el principal.
ax2.set_axis_off()
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# %%
proporciones = train.groupby('Rainfall_range', observed=True)['RainTomorrow'].value_counts(normalize=True).unstack() 

print('Frecuencia relativa de cada grupo\n')
print(frecuencias)
print('\n===================================\n')
print('Proporción de clases de cada grupo\n')
print(proporciones)

# %% [markdown]
# Para hacer el gráfico discretizamos *Rainfall* en 4 rangos de forma que mantengan una frecuencia relativa equilibrada y representativa.
# El primer grupo, que corresponde 0.0 mm de lluvia concentra el 64% de los datos, los demás grupos se reparten los datos equilibradamente, teniendo todos los grupos al menos 10% de los datos.
#
# Podemos observar que la probabilidad de que llueva al dia siguiente es creciente a pasos cada vez mas grandes a medida que que sube el rango de mm de lluvia.
#
# Particularmente la probabilidad de lluvia para el rango `(0.0,1.0]` es de 0.25 y 0.12 para el rango sin lluvia, es decir se duplica, aún asi la variable *RainToday* solo tiene en cuenta los días que cayeron mas de 1mm de agua, es por esto que vamos a quedarnos con la variable *Rainfall* y descartar la variable *RainToday* ya que nos aporta la misma información pero con menos nivel de detalle.
#

# %%
train = train.drop('RainToday', axis=1)

# %% [markdown]
# ### Variable *Sunshine*

# %%
# Crea los bins para Sunshine
bins = [float('-inf'),2.5,5,7.5,9,10,11,12,float('inf')]

intervalos = pd.cut(train['Sunshine'], bins=bins, right=True)

train['Sunshine_range'] = intervalos
# Convierte los intervalos a strings para que Seaborn pueda manejarlos
train['Sunshine_range'] = train['Sunshine_range'].astype(str)

# Asegura que los rangos mantengan el orden
train['Sunshine_range'] = pd.Categorical(
    train['Sunshine_range'],
    categories=[str(interval) for interval in intervalos.cat.categories],
    ordered=True
)

frecuencias = train['Sunshine_range'].value_counts(normalize=True).sort_index()

fig, ax1 = plt.subplots(figsize=(16, 9))
sns.histplot(
    data=train,
    x='Sunshine_range',
    hue='RainTomorrow',
    palette='muted',
    multiple='fill',  # Mostrar proporciones de RainTomorrow en cada bin
    ax=ax1,
)

ax1.set_xlabel('Rango de Sunshine (h)')
ax1.set_ylabel('Proporción de casos que llovió al día siguiente')
ax1.set_title('Distribución de horas de sol registradas y si llovió al día siguiente')

ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

ax1.legend(title='', labels=['Llovió al día siguiente', 'No llovió al dia siguiente'], loc='upper right')

# Segundo eje para la frecuencia relativa
ax2 = ax1.twinx()
ax2.plot(frecuencias.index, frecuencias, color=sns.color_palette('muted')[3], marker='o', label='Frecuencia relativa')
ax2.legend(loc='upper left')

# Oculta el eje y secundario; tiene la misma escala que el principal.
ax2.set_axis_off()
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Variables *Temp9am*, *Temp3pm*, *MinTemp* y *MaxTemp* 

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for i, var in enumerate(['Temp9am', 'Temp3pm', 'MinTemp', 'MaxTemp']):
    sns.boxplot(
        data=train[train['Climate'] == 'Tropical'],
        x=var,
        y='RainTomorrow',
        hue='RainTomorrow',
        palette='muted',
        ax=axes[i // 2, i % 2]
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# Observamos que *MinTemp* presenta una tendencia a temperaturas mínimas mas altas los días previos a que llueva mientras que el resto de las variables por el contrario muestran temperaturas más bajas en los días que llovió al día siguiente. Podríamos sintetizarlo en que los días de lluvia hay menor diferencia entre la mínima y la máxima temperatura a lo largo del día.
#
# Vamos a graficar agrupando por cada tipo de clima para ver si este comportamiento se mantiene.

# %%
# Grafica boxplots comparando variables de temperatura para cada tipo de clima

fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for i, climate in enumerate(train['Climate'].unique()):
    for j, var in enumerate(['Temp9am', 'Temp3pm', 'MinTemp', 'MaxTemp']):
        sns.boxplot(
            data=train[train['Climate'] == climate],
            x=var,
            y='RainTomorrow',
            hue='RainTomorrow',
            order=['No', 'Yes'],
            palette=sns.color_palette('muted')[2*i:2*i+2],
            ax=axes[i, j]
        )
        if j > 0:
            axes[i, j].set_ylabel('')
        elif j == 0:
            axes[i, j].set_ylabel(f'RainTomorrow\n{climate}')
        if i < 2:
            axes[i, j].set_xlabel('')


plt.tight_layout()
plt.show()

# %% [markdown]
# La tendencia a una menor diferencia entre la máxima y la mínima temperatura en los días que llovió al día siguiente. Por lo que vamos a generar una nueva feature *TempDiff*

# %%
train['TempDiff'] = train['MaxTemp'] - train['MinTemp']

test['TempDiff'] = test['MaxTemp'] - test['MinTemp']

# %%
fig, axes = plt.subplots(figsize=(16, 9))
sns.boxplot(
    data=train,
    x='TempDiff',
    y='RainTomorrow',
    hue='RainTomorrow',
    palette='muted',
)

fig.suptitle("Distribución de TempDiff según RainTomorrow")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Variables *Cloud9am* y *Cloud3pm*

# %%
train['RainTomorrowDummy'] = np.where(train['RainTomorrow'] == 'Yes', 1, 0)
proporciones_lluvia = train.dropna().groupby(['Cloud9am', 'Cloud3pm'])['RainTomorrowDummy'].mean().reset_index()

# %%
fig, ax1 = plt.subplots(figsize=(16, 9))

sns.heatmap(
    data=proporciones_lluvia.pivot(index='Cloud9am', columns=('Cloud3pm'), values='RainTomorrowDummy'),
    annot=True,
    fmt=".2f",

)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Variables *Humidity9am* y *Humidity3pm*

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
for i, var in enumerate(['Humidity9am', 'Humidity3pm']):
    sns.boxplot(
        data=train,
        x=var,
        hue='RainTomorrow',
        palette='muted',
        ax=axes[i]
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# Las variables tienen mucha colinealidad y el impacto sobre la target parece reducirse a que simplemente mas humedad mas probabilidad de lluvia al día siguiente. Vamos a quedarnos con Humidity3pm que muestra mayor influencia sobre la target #TODO redaccion

# %%
fig, axes = plt.subplots(3, 2, figsize=(16, 9))
for i, climate in enumerate(train['Climate'].unique()):
    for j, var in enumerate(['Humidity9am', 'Humidity3pm']):
        sns.boxplot(
            data=train[train['Climate'] == climate],
            x=var,
            y='RainTomorrow',
            hue='RainTomorrow',
            order=['No', 'Yes'],
            palette=sns.color_palette('muted')[2*i:2*i+2],
            ax=axes[i, j]
        )
        if j > 0:
            axes[i, j].set_ylabel('')
        elif j == 0:
            axes[i, j].set_ylabel(f'RainTomorrow\n{climate}')
        if i < 2:
            axes[i, j].set_xlabel('')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Variables *Pressure9am* y *Pressure3pm*

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
for i, var in enumerate(['Pressure9am', 'Pressure3pm']):
    sns.boxplot(
        data=train,
        x=var,
        hue='RainTomorrow',
        palette='muted',
        ax=axes[i]
    )

plt.tight_layout()
plt.show()

# %%
fig, ax1 = plt.subplots(figsize=(16, 9))

sns.scatterplot(data=train, x='Pressure9am', y='Pressure3pm', hue='RainTomorrow')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Variables *WindSpeed9am*, *WindSpeed3pm* y *WindGustSpeed*

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
for i, var in enumerate(['WindSpeed9am', 'WindSpeed3pm', 'WindGustSpeed']):
    sns.boxplot(
        data=train,
        x=var,
        hue='RainTomorrow',
        palette='muted',
        ax=axes[i]
    )

plt.tight_layout()
plt.show()

# %%
# Crea los bins para Evaporation
bins = [float('-inf'), 2, 4,6,8,10, float('inf')]

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
    multiple='fill',  # Mostrar proporciones de RainTomorrow en cada bin
    ax=ax1,
)

ax1.set_xlabel('Rango de Evaporation (mm)')
ax1.set_ylabel('Proporción de casos que llovió al día siguiente')
ax1.set_title('Distribución de mm de evaporación registrados y si llovió al día siguiente')

ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

ax1.legend(title='', labels=['Llovió al día siguiente', 'No llovió al dia siguiente'], loc='upper right')

# Segundo eje para la frecuencia relativa
ax2 = ax1.twinx()
ax2.plot(frecuencias.index, frecuencias, color=sns.color_palette('muted')[3], marker='o', label='Frecuencia relativa')
ax2.legend(loc='upper left')

# Oculta el eje y secundario; tiene la misma escala que el principal.
ax2.set_axis_off()
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# %%
fig = plt.figure(figsize=(16,9))
px.scatter_3d(train, x='TempDiff', y='Humidity3pm', z='Rainfall', color='RainTomorrow', width=1600, height=900)

