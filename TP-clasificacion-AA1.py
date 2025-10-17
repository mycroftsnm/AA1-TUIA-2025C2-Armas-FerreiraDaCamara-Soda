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
#     display_name: .venv
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

# %%
# Crea los bins para TEMP
bins = [float('-inf'), 2.5, 5, 7.5, float('inf')]

intervalos = pd.cut(train['TEMP'], bins=bins, right=True)

train['TEMP_range'] = intervalos
# Convierte los intervalos a strings para que Seaborn pueda manejarlos
train['TEMP_range'] = train['TEMP_range'].astype(str)

# Asegura que los rangos mantengan el orden
train['TEMP_range'] = pd.Categorical(
    train['TEMP_range'],
    categories=[str(interval) for interval in intervalos.cat.categories],
    ordered=True
)

frecuencias = train['TEMP_range'].value_counts(normalize=True).sort_index()

fig, ax1 = plt.subplots(figsize=(16, 9))
sns.histplot(
    data=train,
    x='TEMP_range',
    hue='RainTomorrow',
    palette='muted',
    multiple='fill',  # Mostrar proporciones dentro de cada bin
    ax=ax1,
)

ax1.set_xlabel('Rango de Temperatura')
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

# %%
sidney = train[(train['Location'] == 'Sydney') | (train['Location'] == 'SydneyAirport')]

fig, axes = plt.subplots(4, 4, figsize=(20, 18))

for i, var in enumerate(variables_numericas):
    if var == 'Cloud3pm' or var == 'Cloud9am':
        sns.countplot(data=sidney, x=var, hue='Location', palette='muted', ax=axes[i // 4, i % 4])
    else:
        sns.kdeplot(data=sidney, x=var, hue='Location', palette='muted', ax=axes[i // 4, i % 4], common_norm=False)

plt.tight_layout()
plt.show()

# %%
# Crea los bins para Sunshine
bins = [float('-inf'),1, 3,5,7, 10, float('inf')]

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
    multiple='fill',  # Mostrar proporciones dentro de cada bin
    ax=ax1,
)

ax1.set_xlabel('Rango de Sunshine')
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

# %%
train[train.isnull().any(axis=1)]['RainTomorrow'].value_counts()

# %%
train[train.isnull().any(axis=1)]['Climate'].value_counts()

# %%
train.columns

# %%
train[train['Rainfall'].isna()]

# %%
df[df['RainToday'].isna()]['Rainfall'].isna().all()

# %%
df['Sunshine'].isna().sum()

# %%
train[train.isna().sum(axis=1) > 11]

# %%
train[train.isna().sum(axis=1) > 11]['RainTomorrow'].value_counts(dropna=False)

# %%
df['Rainfall'].value_counts(dropna=False)

# %% [markdown]
#

# %%
temp_df = train[['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']].dropna()

# %%
pca = PCA(n_components=1).set_output(transform="pandas")

train['TEMP'] = pca.fit_transform(temp_df)

# %%
pca.explained_variance_ratio_

# %%
fig, ax1 = plt.subplots(figsize=(16, 9))

sns.scatterplot(data=train, x='Pressure9am', y='Pressure3pm', hue='RainTomorrow')

plt.tight_layout()
plt.show()

# %%
fig = plt.figure(figsize=(16,9))
px.scatter_3d(train, x='Temp3pm', y='Humidity3pm', z='Rainfall', color='RainTomorrow', width=1600, height=900)


# %%
# Crea los bins para Cloud
bins = [float('-inf'),1, 3,5,7, 10, float('inf')]

intervalos = pd.cut(train['Cloud3pm'], bins=bins, right=True)

train['Cloud_range'] = intervalos
# Convierte los intervalos a strings para que Seaborn pueda manejarlos
train['Cloud_range'] = train['Cloud_range'].astype(str)

# Asegura que los rangos mantengan el orden
train['Cloud_range'] = pd.Categorical(
    train['Cloud_range'],
    categories=[str(interval) for interval in intervalos.cat.categories],
    ordered=True
)

frecuencias = train['Cloud_range'].value_counts(normalize=True).sort_index()

fig, ax1 = plt.subplots(figsize=(16, 9))
sns.histplot(
    data=train,
    x='Cloud3pm',
    hue='RainTomorrow',
    palette='muted',
    multiple='fill',  # Mostrar proporciones dentro de cada bin
    ax=ax1,
)

ax1.set_xlabel('Rango de Cloud')
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

# %%
proporciones_lluvia = train.dropna().groupby(['Cloud9am', 'Cloud3pm'])['RainTomorrow'].mean().reset_index()

# %%
proporciones_lluvia['RainTomorrow'] = proporciones_lluvia['RainTomorrow'].astype(np.float64)

# %%
fig, ax1 = plt.subplots(figsize=(16, 9))

#sns.scatterplot(data=proporciones_lluvia, x='Cloud9am', y='Cloud3pm', hue='RainTomorrow')
sns.heatmap(
    data=proporciones_lluvia.pivot(index='Cloud9am', columns=('Cloud3pm'), values='RainTomorrow'),
    annot=True,
    fmt=".2f",

)

plt.tight_layout()
plt.show()


# %%
def graficar_prob_lluvia_segun(df, var):
    print(var)
    ax = plt.figure(figsize=(16, 6))
    sns.histplot(
        data=df,
        x=var,
        hue='RainTomorrow',
        palette='muted',
        multiple='fill',  # Mostrar proporciones dentro de cada bin
    )

    

    plt.tight_layout()
    plt.show()

# %%
graficar_prob_lluvia_segun(train, 'Temp9am')

# %%
train['RainTomorrow'].describe()

# %%
ax = plt.figure(figsize=(16, 9))
sns.boxplot(
    data=train,
    y='Temp3pm',
    x='RainTomorrow',
    hue='RainTomorrow',
    palette='muted',
)



plt.tight_layout()
plt.show()
