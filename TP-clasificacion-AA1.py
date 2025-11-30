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
#     display_name: venv_aa
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import shap
import optuna

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    fbeta_score,
    precision_score,
    average_precision_score,
    make_scorer,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from pycaret import classification

import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from custom_transformers import (ClimateTransformer, RainySeasonTransformer,
                          LogTransformer, TempDiffTransformer, 
                          CloudScalerTransformer, WindCyclicalTransformer)

from tensorflow.keras import * 
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

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
df['Date'] = pd.to_datetime(df['Date'])

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
# Comprobamos que efectivamente la clasificación de climas según koppen ayuda a disminuir la multimodalidad al menos sutilmente

# %% [markdown]
# ## Tratado de Outliers

# %% [markdown]
# #### Variable *Rainfall*

# %%
train['Rainfall'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 0.9999])

# %% [markdown]
# Eliminamos los valores mayores al 99.99% de los datos para su posterior imputación ya que presentan un incremento abrupto y son muy pocos datos, aún así estamos quedandonos con valores entre 100 y 185 mm que siguen siendo atípicamente grandes (el 99% de los datos presenta valores inferiores a 37mm) pero sabemos que estos valores son posibles y reales y responden al comportamiento conocido de las lluvías en Australia. 
#
# Después de imputar vamos a aplicar transformación logarítmica para reducir el impacto sobre la media y prevenir overfitting en el modelo de regresión logística.

# %%
train['Rainfall'] = np.where(train['Rainfall'] >= 185, np.nan, train['Rainfall'])
test['Rainfall'] = np.where(test['Rainfall'] >= 185, np.nan, test['Rainfall'])

# %% [markdown]
# #### Variable *Evaporation*

# %%
train['Evaporation'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.9999])

# %% [markdown]
# Eliminamos los valores mayores al 99.99% de los datos para su posterior imputación ya que presentan un incremento abrupto y son muy pocos datos. Aplicamos transformación logarítmica luego de imputar al igual que con *Rainfall*.  

# %%
train['Evaporation'] = np.where(train['Evaporation'] >= 70, np.nan, train['Evaporation'])
test['Evaporation'] = np.where(test['Evaporation'] >= 70, np.nan, test['Evaporation'])

# %% [markdown]
# #### Variable *WindSpeed9am*

# %%
train['WindSpeed9am'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.9999])

# %% [markdown]
# Eliminamos los valores mayores al 99.99% de los datos para su posterior imputación ya que presentan un incremento abrupto y son muy pocos datos.

# %%
train['WindSpeed9am'] = np.where(train['WindSpeed9am'] >= 67, np.nan, train['WindSpeed9am'])

# %% [markdown]
# #### Variable *WindSpeed3pm*

# %%
train['WindSpeed3pm'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.9999])

# %% [markdown]
# #### Variable *WindGustSpeed*

# %%
train['WindGustSpeed'].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99, 0.9999])

# %% [markdown]
# *WindGustSpeed* Representa la máxima velocidad de viento registrada a lo largo de todo el día, por lo que siempre debe ser mayor o igual que *WindSpeed9am* y que *WindSpeed3pm*, vamos a verificar.

# %%
train[train['WindGustSpeed'] < train['WindSpeed9am']]

# %%
train[train['WindGustSpeed'] < train['WindSpeed3pm']]

# %%
# Transforma WindGustSpeed al máximo valor de velocidad de viento.

train['WindGustSpeed'] = np.where(train['WindGustSpeed'] < train['WindSpeed9am'], train['WindSpeed9am'], train['WindGustSpeed'])
train['WindGustSpeed'] = np.where(train['WindGustSpeed'] < train['WindSpeed3pm'], train['WindSpeed3pm'], train['WindGustSpeed'])

test['WindGustSpeed'] = np.where(test['WindGustSpeed'] < test['WindSpeed9am'], test['WindSpeed9am'], test['WindGustSpeed'])
test['WindGustSpeed'] = np.where(test['WindGustSpeed'] < test['WindSpeed3pm'], test['WindSpeed3pm'], test['WindGustSpeed'])

# %% [markdown]
# ## Imputación

# %% [markdown]
# Análisis de variables faltantes e imputación de las mismas.

# %%
train.isnull().sum()

# %%
from scipy.spatial.distance import cdist
def get_closest_location_dict():
    series = []
    for climate, data in australia_coords.groupby('Climate'):
        coords = data[['lat','lon']].values
        dist_matrix = cdist(coords, coords, metric='euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        idxs_min_dist = np.argmin(dist_matrix, axis=1)

        keys = data['location'].values # Ubicacion        
        values = data['location'].iloc[idxs_min_dist].values # Ubicación más cercana

        series.append(pd.Series(values, index=keys))
    
    serie_completa = pd.concat(series)
    # Retorna dict {Ubicacion: Ubicacion mas cercana}
    return serie_completa.to_dict() 


# %%
ubicacion_mas_cercana = get_closest_location_dict()


# %%
def imputar_features(df, features, df_test=None):
    """
    Imputa NaNs en cada feature usando los datos de df.
    Si se pasa df_test imputa sobre ese dataframe.
    """
    # 1. Crear una copia del DataFrame para trabajar de forma segura
    if df_test is not None:
        imputed_df = df_test.copy()
    else:
        imputed_df = df.copy()
    
    for feature in features:
        total_imputados = 0

        variables_direccion_viento = ['WindDir9am', 'WindDir3pm', 'WindGustDir']
        
        df_indexed = df.set_index(['Date', 'Location'])
        if feature not in variables_direccion_viento:
            media_climate_day = df.groupby(['Climate','Date'])[feature].mean()
            medianas_location = df.groupby('Location')[feature].median()
            media_climate = df.groupby(['Climate'])[feature].mean()
            
        nan_rows = imputed_df[imputed_df[feature].isna()]

        for index, row in nan_rows.iterrows():
            
            location = row['Location']
            climate = row['Climate']
            date = row['Date']
            closest_location = ubicacion_mas_cercana[location]
            
            impute_value = np.nan
            
            # 1. Intenta imputar por valor del mismo día en ubicación mas cercana     
            try:
                impute_value = df_indexed.loc[(date, closest_location), feature]
            except KeyError:
                pass

            if pd.isna(impute_value):
                if feature in variables_direccion_viento: # Categóricas
                    # c2. Intenta imputar con el valor de dirección del viento de otra hora del mismo registro
                    for var in variables_direccion_viento:
                        if pd.notna(row[var]):
                            impute_value = row[var]
                            break

                    if pd.isna(impute_value):
                        # c3. Intenta imputar con moda del mismo día para el mismo tipo de clima
                        moda_climate_day_series = df.loc[
                            (df['Climate'] == climate) & (df['Date'] == date), feature
                        ].value_counts()

                        if moda_climate_day_series.empty:
                            impute_value = np.nan
                        else:
                            impute_value = moda_climate_day_series.index[0]

                    if pd.isna(impute_value):
                        # c4. Intenta imputar con moda histórica del tipo de clima
                        moda_climate_series = df.loc[
                            (df['Climate'] == climate) & (df['Date'] == date), feature
                        ].value_counts()

                        if moda_climate_series.empty:
                            impute_value = np.nan
                        else:
                            impute_value = moda_climate_series.index[0]

                else: # Numéricas
                        
                    # n2. Intenta imputar por media del día del mismo tipo de clima
                    impute_value = media_climate_day.get((climate, date))
                    
                    if pd.isna(impute_value):
                        # n3. Intenta imputar por mediana hístórica de la misma ubicación
                        impute_value = medianas_location.get(location)

                    if pd.isna(impute_value):
                        # n4. Intenta imputar por media histórica del mismo tipo de clim
                        impute_value = media_climate.get(climate)
                
            if not pd.isna(impute_value):
                if feature == 'Cloud3pm' or feature == 'Cloud9am':
                    impute_value = round(impute_value)
                imputed_df.loc[index, feature] = impute_value
                total_imputados += 1
            else:
                print('No se pudo imputar')
        print(f'Se imputaron {total_imputados} para la feature {feature}')

    return imputed_df


# %%
variables_a_imputar = variables_numericas + ['WindDir9am', 'WindDir3pm', 'WindGustDir']

train_imputed = imputar_features(train, variables_a_imputar)

# Muestra cuántos NaNs quedan después de la imputación
print("\nConteo de NaNs después de la imputación")
print(train_imputed[variables_a_imputar].isna().sum().sum())

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def comparar_distribucion_final_kde(df_original, df_imputado):
    """
    Compara la distribución de:
    1. Valores originales no faltantes (el esqueleto de la distribución).
    2. Valores finales (el DF imputado completo, incluyendo imputados y originales).
    """
    
    # Prepara el DataFrame Original, setea Origen
    df_inicial = df_original.copy()
    df_inicial['Origen'] = 'Distribución Original'
    
    # Prepara el DataFrame Imputado, setea Origen
    df_final = df_imputado.copy()
    df_final['Origen'] = 'Distribución con datos imputados'
    
    # Concatenar para tener un único conjuntos de datos
    df_combinado = pd.concat([df_inicial, df_final], ignore_index=True)
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 18))

    for i, var in enumerate(variables_numericas):
        if var == 'Cloud3pm' or var == 'Cloud9am':
            sns.countplot(data=df_combinado, x=var, hue='Origen', palette='muted', ax=axes[i // 4, i % 4])
        else:
            sns.kdeplot(data=df_combinado, x=var, hue='Origen', palette='muted', ax=axes[i // 4, i % 4], common_norm=False)

    fig.suptitle('Comparativa de distribuciones de variables numéricas', fontsize=18)

    plt.tight_layout()
    fig.subplots_adjust(top=0.96) # Espacio vertical para el título
    plt.show()


# %%
comparar_distribucion_final_kde(train, train_imputed)

# %% [markdown]
# Las distribuciones se mantienen fieles a los datos originales luego de la imputación, resaltan *Sunshine*, *Cloud9am* y *Cloud3pm* como las features que mas cambiaron su distribución lo cuál tiene sentido debido a que son las features que tenían mayor cantidad de datos faltantes.
#
# Vamos a analizar *Sunshine* separando por clima

# %%
# Prepara el DataFrame Original, setea Origen
df_inicial = train.copy()
df_inicial['Origen'] = 'Distribución Original'

# Prepara el DataFrame Imputado, setea Origen
df_final = train_imputed.copy()
df_final['Origen'] = 'Distribución con datos imputados'

# Concatenar para tener un único conjuntos de datos
df_combinado = pd.concat([df_inicial, df_final], ignore_index=True)

fig, axes = plt.subplots(3, 1, figsize=(16, 9))
for i, climate in enumerate(df_combinado['Climate'].unique()):
    sns.kdeplot(
        data=df_combinado[df_combinado['Climate'] == climate],
        x='Sunshine',
        hue='Origen',
        ax=axes[i],
        common_norm=False,
        palette=sns.color_palette('muted')[2*i:2*i+2],
    )
    axes[i].set_ylabel(climate)

plt.tight_layout()
plt.show()


# %% [markdown]
# Concluímos que la imputación fue buena, ya que las distribuciones se mantienen sin grandes alteraciones.

# %%
test = imputar_features(train, variables_a_imputar, test) # Imputar en test con los datos de train

train = train_imputed

# %% [markdown]
# ## Feature Engineering

# %%
predictoras = [] # Lista de features predictoras

# %%
fig, ax1 = plt.subplots(figsize=(16, 9))

matriz_correlacion = train[variables_numericas].corr()
mascara = np.triu(np.ones_like(matriz_correlacion, dtype=bool))

sns.heatmap(data=matriz_correlacion, ax=ax1, annot=True, vmin=-1, vmax=1, mask=mascara)

fig.suptitle('Matriz de correlación de features continuas')

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

# %%
train["RainTomorrow"].value_counts(normalize=True).round(2)

# %% [markdown]
# Tenemos un gran desbalance entre las clases de la variable objetivo, 78/22

# %%
# Generamos una dummy para RainTomorrow
train['RainTomorrow_dummy'] = np.where(train['RainTomorrow'] == 'Yes', 1, 0)

test['RainTomorrow_dummy'] = np.where(test['RainTomorrow'] == 'Yes', 1, 0)


# %% [markdown]
# ### Variable *Date*

# %% [markdown]
# Generamos la variable *Month* para analizar el comportamiendo de la lluvía a lo largo de los meses.

# %%
train['Month'] = train['Date'].dt.month

# Verificamos la proporción de datos de cada mes separando por tipo de clima

for climate in train['Climate'].unique():
    print(f'\nClima {climate}')
    print(train[train['Climate'] == climate]['Month'].value_counts(normalize=True).sort_index())

# %% [markdown]
# Confirmamos que los datos están uniformemente distribuidos a lo largo de los meses para cada tipo de clima. Continuamos analizando la influencia del mes en la variable objetivo *RainTomorrow*

# %%
train['Month'] = train['Date'].dt.month
test['Month'] = test['Date'].dt.month


fig, axes = plt.subplots(3, 1, figsize=(16, 9))
for i, climate in enumerate(train['Climate'].unique()):
    sns.histplot(
        data=train[train['Climate'] == climate],
        x='Month',
        hue='RainTomorrow',
        ax=axes[i],
        hue_order=['No', 'Yes'],
        multiple='fill',
        discrete=True,
        palette=sns.color_palette('muted')[2*i:2*i+2],
    )
    axes[i].set_ylabel(f'Proporción de RainTomorrow\n{climate}')
    axes[i].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])
    axes[i].set_xticklabels(['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'])

plt.tight_layout()
plt.show()

# %% [markdown]
# Observamos que los climas *Tropical* y *Temperate* aumentan considerablemente la proporción de dias que llovió al dia siguiente, entre diciembre y marzo *Tropical* y entre junio y septiembre *Temperate*.
# El clima árido mantiene una proporción baja a lo largo del año, levemente más baja en los primeros 4 meses del año.
#
# Como los períodos de mayor actividad suceden en momentos distintos para cada clima no vamos a codificar el mes usando seno y coseno, en cambio vamos a generar la variable *RainySeason* para marcar el cuatrimestre de mayor lluvia para cada clima.

# %%
meses_tropical = set([12,1,2,3])
meses_temperate = set([6,7,8,9])

train['RainySeason'] = np.where(
    ((train['Climate'] == 'Tropical') & train['Month'].isin(meses_tropical)) |
    ((train['Climate'] == 'Temperate') & train['Month'].isin(meses_temperate)),
    1,
    0)

test['RainySeason'] = np.where(
    ((test['Climate'] == 'Tropical') & test['Month'].isin(meses_tropical)) |
    ((test['Climate'] == 'Temperate') & test['Month'].isin(meses_temperate)),
    1,
    0) 

predictoras.append('RainySeason')

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
# Vemos que la probabilidad de que llueva se triplica pasando de 15,1% a 46,2%. Sin embargo no deja de ser siempre más probable que no llueva a que sí lo haga, sin importar si llovió el día anterior.
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
# El primer grupo, que corresponde 0.0 mm de lluvia concentra el 63% de los datos, los demás grupos se reparten los datos equilibradamente, teniendo todos los grupos al menos 10% de los datos.
#
# Podemos observar que la probabilidad de que llueva al dia siguiente es creciente a pasos cada vez mas grandes a medida que que sube el rango de mm de lluvia.
#
# Particularmente la probabilidad de lluvia para el rango `(0.0,1.0]` es de 0.25, duplicando el valor 0.13 del rango sin lluvia, aún asi la variable *RainToday* solo tiene en cuenta los días que cayeron mas de 1mm de agua, es por esto que vamos a quedarnos con la variable *Rainfall* y descartar la variable *RainToday* ya que nos aporta la misma información pero con menos nivel de detalle.
#

# %%
train['Rainfall_log'] = np.log1p(train['Rainfall'])
test['Rainfall_log'] = np.log1p(test['Rainfall'])

predictoras.append('Rainfall_log')

# %% [markdown]
# ### Variable *Evaporation*

# %%
# Crea los bins para Evaporation
bins = [float('-inf'), 2.5, 5,7.5, float('inf')]

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
proporciones = train.groupby('Evaporation_range', observed=True)['RainTomorrow'].value_counts(normalize=True).unstack() 

print('Frecuencia relativa de cada grupo\n')
print(frecuencias)
print('\n===================================\n')
print('Proporción de clases de cada grupo\n')
print(proporciones)

# %% [markdown]
# El gráfico se construyó con lógica análoga al de *RainFall*. Observamos que a medida que aumenta el rango de evaporación dismininuye gradualmente la propoción de casos en los que llovió al día siguiente. Particularmente para valores de *Evaporation* mayores a 7.5, solo en el 14% de los casos llovió al día siguiente. Vamos a considerar esta variable para nuestro modelo, teniendo en cuenta la transformación logarítmica.

# %%
train['Evaporation_log'] = np.log1p(train['Evaporation'])
test['Evaporation_log'] = np.log1p(test['Evaporation'])

predictoras.append('Evaporation_log')

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
ax1.set_title('Distribución de horas de sol en el día y si llovió al día siguiente')

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
# El gráfico se construyó con lógica análoga al de *RainFall* y *Evaporation*. Observamos que a medida que aumenta el rango de horas de sol dismininuye consistentemente la propoción de casos en los que llovió al día siguiente. Vamos a tener en cuenta esta feature para nuestro modelo.

# %%
predictoras.append('Sunshine')

# %% [markdown]
# ### Variables *Temp9am*, *Temp3pm*, *MinTemp* y *MaxTemp* 

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for i, var in enumerate(['Temp9am', 'Temp3pm', 'MinTemp', 'MaxTemp']):
    sns.boxplot(
        data=train,
        x=var,
        y='RainTomorrow',
        hue='RainTomorrow',
        palette='muted',
        ax=axes[i // 2, i % 2]
    )

fig.suptitle("Distribución de varíables de temperatura según RainTomorrow")

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

fig.suptitle("Distribución de varíables de temperatura según RainTomorrow por tipo de clima")

plt.tight_layout()
plt.show()

# %% [markdown]
# La tendencia a una menor amplitud térmica en los días que llovió al día siguiente se mantiene presente para todos los climas. Por lo que vamos a generar una nueva feature *TempDiff*

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
# La feature generada muestra una diferencía marcada entre los casos que llovío al día siguiente y los que no. Por ejemplo la mediana de la clase 'Yes' esta fuera de la caja de la clase 'No'. Vamos a considerar a está feature como buena predictora para nuestro modelo,  también vamos a incluir a *MinTemp*.

# %%
predictoras.append('MinTemp')
predictoras.append('TempDiff')

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

fig.suptitle("Probabilidad de RainTomorrow según Cloud9am y Cloud3pm")

plt.tight_layout()
plt.show()

# %% [markdown]
# Observamos que la probabilidad de que llueva al día siguiente aumenta a medida que aumenta la nubosidad general en el día. Sin embargo el nivel de nubosidad de las 3pm parece ser mucho más determinista para la probabilidad de llueva el próximo día que el de las 9am, por ejemplo:
#
# Sin importar que tan nublado haya estado el cielo a las 9am, si a las 3pm el cielo estuvo despejado en el 99.99% de los casos no llovió al día siguiente.
# Al mismo tiempo, si a las 3pm el cielo estuvo completamente nublado, la proporción de casos en los que llovió al día siguiente se incrementa considerablemente para todos los valores de nubosidad de las 9am.
#
# En principio vamos a mantener ambas variables y probar el desempeño del modelo, para luego compararlo eliminando Cloud9am.

# %%
predictoras.append('Cloud9am')
predictoras.append('Cloud3pm')

# %% [markdown]
# ### Variables *Humidity9am* y *Humidity3pm*

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
for i, var in enumerate(['Humidity9am', 'Humidity3pm']):
    sns.boxplot(
        data=train,
        x=var,
        y='RainTomorrow',
        hue='RainTomorrow',
        palette='muted',
        ax=axes[i]
    )

fig.suptitle("Distribución de variables de humedad según RainTomorrow")

plt.tight_layout()
plt.show()

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

fig.suptitle("Distribución de variables de humedad según RainTomorrow por clima")

plt.tight_layout()
plt.show()

# %% [markdown]
# Las variables presentan una alta colinealidad, en general los días con mayor humedad presentan una mayor de proporción de casos en los que lovió al día siguiente, este comportamiento se mantiene en todos los tipos de clima. Vamos a quedarnos con *Humidity3pm* para reducir la multicolinealidad del módelo ya que presenta una influencia más marcada. Update: Nos quedamos con ambas ya que mejorá el desempeño del modelo.

# %%
predictoras.append('Humidity3pm')
predictoras.append('Humidity9am')

# %% [markdown]
# ### Variables *Pressure9am* y *Pressure3pm*

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
for i, var in enumerate(['Pressure9am', 'Pressure3pm']):
    sns.boxplot(
        data=train,
        x=var,
        y='RainTomorrow',
        hue='RainTomorrow',
        palette='muted',
        ax=axes[i]
    )

fig.suptitle("Distribución de variables de presión según RainTomorrow")

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(3, 2, figsize=(16, 9))
for i, climate in enumerate(train['Climate'].unique()):
    for j, var in enumerate(['Pressure9am', 'Pressure3pm']):
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

fig.suptitle("Distribución de variables de presión según RainTomorrow por clima")

plt.tight_layout()
plt.show()

# %% [markdown]
# Idem variables de humedad. Nos quedamos con *Pressure3pm*. Update: Nos quedamos con ambas porque mejora el desempeño del modelo.

# %%
predictoras.append('Pressure3pm')
predictoras.append('Pressure9am')

# %% [markdown]
# ### Variables *WindSpeed9am*, *WindSpeed3pm* y *WindGustSpeed*

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
for i, var in enumerate(['WindSpeed9am', 'WindSpeed3pm', 'WindGustSpeed']):
    sns.boxplot(
        data=train,
        x=var,
        y='RainTomorrow',
        hue='RainTomorrow',
        palette='muted',
        ax=axes[i]
    )

fig.suptitle("Distribución de WindSpeed9am, WindSpeed3pm y WindGustSpeed según RainTomorrow")

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(3, 3, figsize=(16, 9))
for i, climate in enumerate(train['Climate'].unique()):
    for j, var in enumerate(['WindSpeed9am', 'WindSpeed3pm', 'WindGustSpeed']):
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

fig.suptitle("Distribución de variables de velocidad de viento según RainTomorrow por clima")

plt.tight_layout()
plt.show()

# %% [markdown]
# Idem variables de humedad y de temperatura. Nos quedamos con *WindGustSpeed*

# %%
predictoras.append('WindSpeed9am')
predictoras.append('WindSpeed3pm')
predictoras.append('WindGustSpeed')

# %% [markdown]
# ### Variables *WindDir9am*, *WindDir3pm* y *WindGustDir*

# %%
fig, axes = plt.subplots(3, 1, figsize=(16, 9))
for i, var in enumerate(['WindDir9am', 'WindDir3pm', 'WindGustDir']):
    sns.histplot(
        data=train,
        x=var,
        hue='RainTomorrow',
        palette='muted',
        multiple='fill',
        ax=axes[i]
    )

fig.suptitle("Distribución de WindDir9am, WindDir3pm y WindGustDir según RainTomorrow")

plt.tight_layout()
plt.show()

# %% [markdown]
# Estas variables son complejas ya que dependen de cada ubicación en particular, vamos a encodearlas cíclicamente usando seno y coseno.

# %%
dir_angulos = {
    'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5, 'E': 90.0,
    'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5, 'S': 180.0, 'SSW': 202.5,
    'SW': 225.0, 'WSW': 247.5, 'W': 270.0, 'WNW': 292.5, 'NW': 315.0,
    'NNW': 337.5
}

def encodear_direccion_viento(df, var):
    angulos = df[var].map(dir_angulos)
    df[f'{var}_sin'] = np.sin(angulos * 2 * np.pi / 360)
    df[f'{var}_cos'] = np.cos(angulos * 2 * np.pi / 360)

    return df


for var in ['WindDir9am', 'WindDir3pm', 'WindGustDir']:
    train = encodear_direccion_viento(train, var)
    test = encodear_direccion_viento(test, var)
    predictoras.append(f'{var}_sin')
    predictoras.append(f'{var}_cos')

# %%

fig, ax1 = plt.subplots(figsize=(16, 9))

predictoras_a_graficar = predictoras.copy()
predictoras_a_graficar.remove('WindGustDir_sin')
predictoras_a_graficar.remove('WindGustDir_cos')
predictoras_a_graficar.remove('WindDir9am_sin')
predictoras_a_graficar.remove('WindDir9am_cos')
predictoras_a_graficar.remove('WindDir3pm_sin')
predictoras_a_graficar.remove('WindDir3pm_cos')

matriz_correlacion = train[predictoras_a_graficar].corr()
mascara = np.triu(np.ones_like(matriz_correlacion, dtype=bool))

sns.heatmap(data=matriz_correlacion, ax=ax1, annot=True, vmin=-1, vmax=1, mask=mascara)

plt.tight_layout()
plt.show()

# %% [markdown]
# # PreTrain

# %% [markdown]
# ## Estandarización 

# %% [markdown]
# Estandarizamos todas las features númericas continuas que consideramos predictoras

# %%
predictoras_continuas = predictoras.copy()
predictoras_continuas.remove('Cloud9am')
predictoras_continuas.remove('Cloud3pm')
predictoras_continuas.remove('RainySeason')
print(f"Variables a estandarizar: {predictoras_continuas}")

# %% [markdown]
# Vamos a escalar las features *Cloud9am* y *Cloud3pm*, que son variables númericas discretas, usando la técnica de MinMaxScaler

# %%
train['Cloud9am'] = train['Cloud9am'] / 8
train['Cloud3pm'] = train['Cloud3pm'] / 8

test['Cloud9am'] = test['Cloud9am'] / 8
test['Cloud3pm'] = test['Cloud3pm'] / 8


# %%
scaler = StandardScaler()
scaler.fit(train[predictoras_continuas])

train[predictoras_continuas] = scaler.transform(train[predictoras_continuas])
test[predictoras_continuas] = scaler.transform(test[predictoras_continuas])

# %% [markdown]
# ## Generación de variables dummy's

# %% [markdown]
# Transformamos la variable categórica *Climate* mediante OneHotEncoding para incorporarlas al modelo.

# %%
train['ClimateArid'] = np.where(train['Climate'] == 'Arid', 1, 0)
test['ClimateArid'] = np.where(test['Climate'] == 'Arid', 1, 0)

train['ClimateTropical'] = np.where(train['Climate'] == 'Tropical', 1, 0)
test['ClimateTropical'] = np.where(test['Climate'] == 'Tropical', 1, 0)

predictoras.append('ClimateArid')
predictoras.append('ClimateTropical')

# %%
print(f'Vamos a utilizar las siguientes variables para entrenar el modelo:\n{predictoras}')


# %% [markdown]
# # Modelado

# %%
def f2_score(y_true, y_pred, **kwargs):
    return fbeta_score(y_true, y_pred, beta=2, pos_label=1)


# %%
def evaluar_modelo(y_true, y_pred, y_pred_proba, nombre_modelo):
    """
    Evalúa un modelo y retorna un diccionario con las métricas
    """
    resultados = {
        'Modelo': nombre_modelo,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': classification_report(y_true, y_pred, output_dict=True)['1']['precision'],
        'Recall': classification_report(y_true, y_pred, output_dict=True)['1']['recall'],
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba),
        'F2-Score': f2_score(y_true, y_pred)
    }
    
    return resultados

def graficar_matriz_confusion(y_true, y_pred, titulo):
    """
    Grafica la matriz de confusión
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')
    ax.set_title(f'Matriz de Confusión - {titulo}')
    ax.set_xticklabels(['No llueve', 'Llueve'])
    ax.set_yticklabels(['No llueve', 'Llueve'])
    plt.tight_layout()
    plt.show()


# %%
x_train = train[predictoras]
y_train = train['RainTomorrow_dummy']

x_test = test[predictoras]
y_test = test['RainTomorrow_dummy']

# %% [markdown]
# Para optimizar los hiperparámetros elegimos maximizar la métrica F2_score,
# El F2-Score es una variante del F-beta score donde β = 2, lo que significa que:
#
# `F2 = (1 + 2²) × (Precision × Recall) / (2² × Precision + Recall)`
#
# `F2 = 5 × (Precision × Recall) / (4 × Precision + Recall)`
#
# Esto implica que el Recall tiene 2 veces más peso que la Precision, lo que en nuestro caso es ideal porque decidimos:
# - minimizar Falsos Negativos (no predecir lluvia cuando sí llueve)
# - priorizar True Positives (detectar correctamente los días de lluvia)
# - por consecuencia la idea es ser más permisivos con los Falsos Positivos (predecir lluvia de más que de menos)
#
# ### Comentario sobre el umbral.
# No haremos optimización del umbral dentro de esta primera optimización de hiperparámetros por cuestiones de eficiencia.
# Para encontrar el umbral es posible hacerlo con los modelos ya entrenados. Es decir, el umbral no afecta el entrenamiento.
# Por lo tanto, el enfoque será primero optimizar C y penalty (con Optuna) y posteriormente con grid search encontrar el umbral óptimo
#
# ### Hiperparámetros a optimizar
# 1. C (Regularization Strength)
# ```python
#   C = trial.suggest_float('C', 1e-4, 1e4, log=True)
# ```
# - Parámetro de regularización inverso: valores más pequeños = regularización más fuerte
# - Controla el trade-off entre ajustar bien los datos de entrenamiento vs. generalizar
#
# **Rango explorado**:
#
# `1e-4` a `1e4` en escala logarítmica: C puede variar varios órdenes de magnitud, por eso se utiliza escala logarítmica.
#
# **Implicaciones**:
#
# - C pequeño (`1e-4`): modelo simple, puede subajustar (underfitting), menos propenso a overfitting
# - C grande (`1e4`): modelo complejo, puede sobreajustar (overfitting), se ajusta más a los datos de entrenamiento
#
# Controla directamente cuánto el modelo puede aprender de los datos.
# Relacionándolo con la teoría viste de regularización, C se relaciona con el valor α directamente: $\alpha = \frac{1}{C}$.
#
#
# 2. penalty (Tipo de regularización)
# ```python
#    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
#    l1_ratio = trial.suggest_float('l1_ratio', 0, 1) if penalty == 'elasticnet' else None
# ```
# Define el tipo de norma usada en la penalización
# |Penalty|Nombre|Efecto|
# |---|---|---|
# |`L1`|Lasso|Elimina variables que no son importantes|
# |`L2`|Ridge|Reduce los coeficientes pero no los elimina completamente|
# |`elasticnet`|ElasticNet|Combinación de Lasso y Ridge|
#
# Si la regularización elegida es ElasticNet, se debe optimizar tambien `l1_ratio`: si es bajo se acerca más a la regularización de L1,
# si es alto tiende más a L2. Cuando está cercano a 0.5 corresponde a un balance de ambas.
# ### Hiperparámetros fijos (que no se optimizarán)
# - `solver=saga`:     Algotitmo de optimización
# - `max_iter=1000`:   Número máximo de iteraciones para convergencia
# - `random_state=42`: Semilla para reproducibilidad
# - `n_jobs=-1`:       Usa todos los cores disponibles del CPU
# %% [markdown]
# ## Modelo 1: Regresión Logística sin balanceo
# %%
# Creamos un objeto "scorer" para cross_val_score. Este será utilizado por todas las funciones objective.
# fbeta_score con beta=2 y promedio ponderado.
f2_scorer = make_scorer(fbeta_score, beta=2, average='weighted', pos_label=1)

def objective_lr_base(trial, X_data, y_data):
    """
    Función objective para el Modelo 1 (base, sin balanceo).
    Maximiza el F2-Score
    """
    
    # a. Definición del espacio de búsqueda de hiperparámetros
    C = trial.suggest_float('C', 1e-4, 1e4, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1) if penalty == 'elasticnet' else None # Se usa solo si penalty es 'elasticnet'
    
    # b. Modelo con los parámetros trial
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        l1_ratio=l1_ratio,
        solver='saga',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    # c. Evaluación del modelo usando Cross-Validation (sobre x_train, y_train)

    # Usamos cv=5 y 'f2_weighted' como métrica
    scores = cross_val_score(
        model, 
        X_data, 
        y_data, 
        cv=5, 
        scoring=f2_scorer, 
        n_jobs=-1
    )
    metric_value = np.mean(scores)
    
    # d. Devuelve la métrica a maximizar
    return metric_value

# Ejecución de Optuna 

print("Optimizando Modelo 1: Regresión Logística sin balanceo...")

# Queremos MAXIMIZAR el F2-Score
study = optuna.create_study(direction='maximize')

# Pasamos los datos de entrenamiento (x_train, y_train) a la función objective
study.optimize(
    lambda trial: objective_lr_base(trial, x_train, y_train),
    n_trials=25, # Número de intentos de optimización 
    show_progress_bar=True
)

print("\nOptimización completada.")

# Entrenamiento y Evaluación Final del Modelo 1 

# a. Recuperar los mejores hiperparámetros encontrados
best_optuna_params = study.best_params
print(f"Mejor F2-Score (promedio en CV): {study.best_value:.5f}")
print(f"Mejores hiperparámetros: {best_optuna_params}")

# b. Combinar parámetros fijos y optimizados
final_model_params = {
    **best_optuna_params,
    'solver': 'saga',
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1
}

# c. Entrenar el modelo definitivo con esos parámetros
# Se entrena sobre TODOS los datos de x_train
print("\nEntrenando modelo final optimizado sobre x_train...")
best_optuna_model_1 = LogisticRegression(**final_model_params)
best_optuna_model_1.fit(x_train, y_train)

# d. Evaluar el modelo final en el set de PRUEBA (x_test, y_test)
print("Evaluando modelo final sobre x_test...")
y_pred_unbalanced_final = best_optuna_model_1.predict(x_test)

# Calcular métricas finales
optuna_f2_unbalanced_final = fbeta_score(y_test, y_pred_unbalanced_final, beta=2, pos_label=1)

print("-" * 30)
print(f"F2-Score FINAL (en test): {optuna_f2_unbalanced_final:.5f}")
print("\nReporte de Clasificación Final (en test):")
print(classification_report(y_test, y_pred_unbalanced_final, target_names=['No llueve', 'Llueve']))

y_pred_proba_unbalanced_final = best_optuna_model_1.predict_proba(x_test)[:, 1]
resultados_unbalanced_final = evaluar_modelo(y_test, y_pred_unbalanced_final, y_pred_proba_unbalanced_final, 'Sin balanceo')

# aplicamos la función para graficar la matriz de confusión anteriormente definida
graficar_matriz_confusion(y_test, y_pred_unbalanced_final, 'Modelo 1 - LR Base (Optimizado sin balancear)')
# %% [markdown]
# ## Modelo 2: Regresión Logística con balanceo de clase
# %%
def objective_lr_balanced(trial, X_data, y_data):
    """
    Función objective para el Modelo 2 (con balanceo).
    """
    
    # a. Definición del espacio de búsqueda de hiperparámetros
    C = trial.suggest_float('C', 1e-4, 1e4, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1) if penalty == 'elasticnet' else None
    
    # b. Modelo con los parámetros trial
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        l1_ratio=l1_ratio,
        solver='saga',
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced' # se agrega en este caso para balancear las clases
    )
    
    # c. Evaluación del modelo usando Cross-Validation (sobre x_train, y_train)
    scores = cross_val_score(
        model, 
        X_data, 
        y_data, 
        cv=5, 
        scoring=f2_scorer, 
        n_jobs=-1
    )
    metric_value = np.mean(scores)

    # d. Devuelve la métrica a maximizar
    return metric_value

# Ejecución de Optuna Study

print("Optimizando Modelo 2: Regresión Logística con balanceo...")

study = optuna.create_study(direction='maximize')

# Pasamos los datos de entrenamiento (x_train, y_train) a la función objective
study.optimize(
    lambda trial: objective_lr_balanced(trial, x_train, y_train),
    n_trials=25, # Número de intentos de optimización
    show_progress_bar=True
)

print("\nOptimización completada.")

# Entrenamiento y Evaluación Final del Modelo 2

# a. Recuperar los mejores hiperparámetros encontrados
best_optuna_params = study.best_params
print(f"Mejor F2-Score (promedio en CV): {study.best_value:.5f}")
print(f"Mejores hiperparámetros: {best_optuna_params}")

# b. Combinar parámetros fijos y optimizados
final_model_params = {
    **best_optuna_params,
    'solver': 'saga',
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced' # se agrega tanbién acá 
}

# c. Entrenar el modelo definitivo con esos parámetros
# Se entrena sobre TODOS los datos de x_train
print("\nEntrenando modelo final optimizado sobre x_train...")
best_optuna_model_2 = LogisticRegression(**final_model_params)
best_optuna_model_2.fit(x_train, y_train)

# d. Evaluar el modelo final en el set de PRUEBA (x_test, y_test)
print("Evaluando modelo final sobre x_test...")
y_pred_balanced_final = best_optuna_model_2.predict(x_test)

# Calcular métricas finales
optuna_f2_balanced_final = fbeta_score(y_test, y_pred_balanced_final, beta=2, pos_label=1)

print("-" * 30)
print(f"F2-Score FINAL (en test): {optuna_f2_balanced_final:.5f}")
print("\nReporte de Clasificación Final (en test):")
print(classification_report(y_test, y_pred_balanced_final, target_names=['No llueve', 'Llueve']))

y_pred_proba_balanced_final = best_optuna_model_2.predict_proba(x_test)[:, 1]
resultados_balanced_final = evaluar_modelo(y_test, y_pred_balanced_final, y_pred_proba_balanced_final, 'Class Weight Balanced')

# mc
graficar_matriz_confusion(y_test, y_pred_balanced_final, 'Modelo 2 - LR Optimizado con clases balanceadas')

# %% [markdown]
# ## Modelo 3: Regresión Lógistica con SMOTE
# %%
# Aplicar SMOTE al conjunto de entrenamiento
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

def objective_lr_smote(trial, X_data, y_data):
    """
    Función objective para el Modelo 3 (SMOTE).
    """
    
    # a. Definición del espacio de búsqueda de hiperparámetros
    C = trial.suggest_float('C', 1e-4, 1e4, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1) if penalty == 'elasticnet' else None
    
    # b. Modelo con los parámetros trial
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        l1_ratio=l1_ratio,
        solver='saga',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    # c. Evaluación del modelo usando Cross-Validation (sobre x_train, y_train)
    scores = cross_val_score(
        model, 
        X_data, 
        y_data, 
        cv=5, 
        scoring=f2_scorer, 
        n_jobs=-1
    )
    metric_value = np.mean(scores)
    
    # d. Devuelve la métrica a maximizar
    return metric_value

# Ejecución de Optuna Study

print("Optimizando Modelo 3: Regresión Logística (con balanceo SMOTE)...")

study = optuna.create_study(direction='maximize')

# Pasamos los datos de entrenamiento (x_train_smote, y_train_smote) a la función objective
study.optimize(
    lambda trial: objective_lr_smote(trial, x_train_smote, y_train_smote),
    n_trials=25, # Número de intentos de optimización
    show_progress_bar=True
)

print("\nOptimización completada.")

# Entrenamiento y Evaluación Final del Modelo 3 SMOTE

# a. Recuperar los mejores hiperparámetros encontrados
best_optuna_params = study.best_params
print(f"Mejor F2-Score (promedio en CV): {study.best_value:.5f}")
print(f"Mejores hiperparámetros: {best_optuna_params}")

# b. Combinar parámetros fijos y optimizados
final_model_params = {
    **best_optuna_params,
    'solver': 'saga',
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1
}

# c. Entrenar el modelo definitivo con esos parámetros
# Se entrena sobre TODOS los datos de x_train_smote
print("\nEntrenando modelo final optimizado sobre x_train_smote...")
best_optuna_model_3 = LogisticRegression(**final_model_params)
best_optuna_model_3.fit(x_train_smote, y_train_smote)

# d. Evaluar el modelo final en el set de PRUEBA (x_test, y_test)
print("Evaluando modelo final sobre x_test...")
y_pred_smote_final = best_optuna_model_3.predict(x_test)

# Calcular métricas finales
optuna_f2_smote_final = fbeta_score(y_test, y_pred_smote_final, beta=2, pos_label=1)

print("-" * 30)
print(f"F2-Score FINAL (en test): {optuna_f2_smote_final:.5f}")
print("\nReporte de Clasificación Final (en test):")
print(classification_report(y_test, y_pred_smote_final, target_names=['No llueve', 'Llueve']))

y_pred_proba_smote_final = best_optuna_model_3.predict_proba(x_test)[:, 1]
resultados_smote_final = evaluar_modelo(y_test, y_pred_smote_final, y_pred_proba_smote_final, 'SMOTE')

# mc
graficar_matriz_confusion(y_test, y_pred_smote_final, 'Modelo 3 - SMOTE')

# %% [markdown]
# ## Modelo 4: Regresión Logística con Random Under-Samplig
# %%
# Aplicar Random Under-Sampling
rus = RandomUnderSampler(random_state=42)
x_train_rus, y_train_rus = rus.fit_resample(x_train, y_train)

print(f"Distribución después de Under-Sampling:")
print(pd.Series(y_train_rus).value_counts())

def objective_lr_rus(trial, X_data, y_data):
    """
    Función objective para el Modelo 4 (Random Under-Sampling).
    """
    
    # a. Definición del espacio de búsqueda de hiperparámetros
    C = trial.suggest_float('C', 1e-4, 1e4, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1) if penalty == 'elasticnet' else None
    
    # b. Modelo con los parámetros trial
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        l1_ratio=l1_ratio,
        solver='saga',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    # c. Evaluación del modelo usando Cross-Validation (sobre x_train, y_train)
    scores = cross_val_score(
        model, 
        X_data, 
        y_data, 
        cv=5, 
        scoring=f2_scorer, 
        n_jobs=-1
    )
    metric_value = np.mean(scores)
    
    # d. Devuelve la métrica a maximizar
    return metric_value

# Ejecución de Optuna Study

print("Optimizando Modelo 4: Regresión Logística (Under-Sampling)...")

study = optuna.create_study(direction='maximize')

# Pasamos los datos de entrenamiento (x_train_rus, y_train_rus) a la función objective
study.optimize(
    lambda trial: objective_lr_rus(trial, x_train_rus, y_train_rus),
    n_trials=25, # Número de intentos de optimización
    show_progress_bar=True
)

print("\nOptimización completada.")

# Entrenamiento y Evaluación Final del Modelo 4 Random Under-Sampling

# a. Recuperar los mejores hiperparámetros encontrados
best_optuna_params = study.best_params
print(f"Mejor F2-Score (promedio en CV): {study.best_value:.5f}")
print(f"Mejores hiperparámetros: {best_optuna_params}")

# b. Combinar parámetros fijos y optimizados
final_model_params = {
    **best_optuna_params,
    'solver': 'saga',
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1
}

# c. Entrenar el modelo definitivo con esos parámetros
# Se entrena sobre TODOS los datos de x_train_rus
print("\nEntrenando modelo final optimizado sobre x_train_rus...")
best_optuna_model_4 = LogisticRegression(**final_model_params)
best_optuna_model_4.fit(x_train_rus, y_train_rus)

# d. Evaluar el modelo final en el set de PRUEBA (x_test, y_test)
print("Evaluando modelo final sobre x_test...")
y_pred_rus_final = best_optuna_model_4.predict(x_test)

# Calcular métricas finales
optuna_f2_rus_final = fbeta_score(y_test, y_pred_rus_final, beta=2, pos_label=1)

print("-" * 30)
print(f"F2-Score FINAL (en test): {optuna_f2_rus_final:.5f}")
print("\nReporte de Clasificación Final (en test):")
print(classification_report(y_test, y_pred_rus_final, target_names=['No llueve', 'Llueve']))

y_pred_proba_rus_final = best_optuna_model_4.predict_proba(x_test)[:, 1]
resultados_rus_final = evaluar_modelo(y_test, y_pred_rus_final, y_pred_proba_rus_final, 'Random Under-Sampling')

# mc
graficar_matriz_confusion(y_test, y_pred_rus_final, 'Modelo 4 - Random Under-Sampling')



# %%
# Aplicar Random Over-Sampling
ros = RandomOverSampler(random_state=42)
x_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)

print(f"Distribución después de Over-Sampling:")
print(pd.Series(y_train_ros).value_counts())

def objective_lr_ros(trial, X_data, y_data):
    """
    Función objective para el Modelo 5 (Random Over-Sampling).
    """
    
    # a. Definición del espacio de búsqueda de hiperparámetros
    C = trial.suggest_float('C', 1e-4, 1e4, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1) if penalty == 'elasticnet' else None
    
    # b. Modelo con los parámetros trial
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        l1_ratio=l1_ratio,
        solver='saga',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    # c. Evaluación del modelo usando Cross-Validation (sobre x_train, y_train)

    # Usamos cv=5 para robustez y 'f2_weighted' como métrica
    scores = cross_val_score(
        model, 
        X_data, 
        y_data, 
        cv=5, 
        scoring=f2_scorer, 
        n_jobs=-1
    )
    metric_value = np.mean(scores)
    
    # d. Devuelve la métrica a maximizar
    return metric_value

# Ejecución de Optuna Study

print("Optimizando Modelo 5: Regresión Logística (Over-Sampling)...")

study = optuna.create_study(direction='maximize')

# Pasamos los datos de entrenamiento (x_train_ros, y_train_ros) a la función objective
study.optimize(
    lambda trial: objective_lr_ros(trial, x_train_ros, y_train_ros),
    n_trials=25, # Número de intentos de optimización
    show_progress_bar=True
)

print("\nOptimización completada.")

# Entrenamiento y Evaluación Final del Modelo 5 Random Over-Sampling

# a. Recuperar los mejores hiperparámetros encontrados
best_optuna_params = study.best_params
print(f"Mejor F2-Score (promedio en CV): {study.best_value:.5f}")
print(f"Mejores hiperparámetros: {best_optuna_params}")

# b. Combinar parámetros fijos y optimizados
final_model_params = {
    **best_optuna_params,
    'solver': 'saga',
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1
}

# c. Entrenar el modelo definitivo con esos parámetros
# Se entrena sobre TODOS los datos de x_train_ros
print("\nEntrenando modelo final optimizado sobre x_train_ros...")
best_optuna_model_5 = LogisticRegression(**final_model_params)
best_optuna_model_5.fit(x_train_ros, y_train_ros)

# d. Evaluar el modelo final en el set de PRUEBA (x_test, y_test)
print("Evaluando modelo final sobre x_test...")
y_pred_ros_final = best_optuna_model_5.predict(x_test)

# Calcular métricas finales
optuna_f2_ros_final = fbeta_score(y_test, y_pred_ros_final, beta=2, pos_label=1)

print("-" * 30)
print(f"F2-Score FINAL (en test): {optuna_f2_ros_final:.5f}")
print("\nReporte de Clasificación Final (en test):")
print(classification_report(y_test, y_pred_ros_final, target_names=['No llueve', 'Llueve']))

y_pred_proba_ros_final = best_optuna_model_5.predict_proba(x_test)[:, 1]
resultados_ros_final = evaluar_modelo(y_test, y_pred_ros_final, y_pred_proba_ros_final, 'Random Over-Sampling')

# mc
graficar_matriz_confusion(y_test, y_pred_ros_final, 'Modelo 5 - Random Over-Sampling')

# %%
# Crear DataFrame con todos los resultados
df_resultados = pd.DataFrame([
    resultados_unbalanced_final,
    resultados_balanced_final,
    resultados_smote_final,
    resultados_rus_final,
    resultados_ros_final
])

print("\nCOMPARACIÓN DE MODELOS")
print(df_resultados.to_string(index=False))

# %%
fig, axes = plt.subplots(5, 1, figsize=(16, 18))

metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

for idx, metrica in enumerate(metricas):
    ax = axes[idx]
    sns.barplot(data=df_resultados, hue='Modelo', x='Modelo', y=metrica, palette='muted', ax=ax)
    ax.set_title(f'{metrica}', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(metrica, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel('')
    # Agregar valores sobre las barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)


fig.suptitle('Comparación de Métricas por Modelo', fontsize=16) 
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Curvas ROC

# %%
fig, ax = plt.subplots(figsize=(12, 8))

modelos_predicciones = [
    ('Sin balanceo', y_pred_proba_unbalanced_final),
    ('Class Weight Balanced', y_pred_proba_balanced_final),
    ('SMOTE', y_pred_proba_smote_final),
    ('Random Under-Sampling', y_pred_proba_rus_final),
    ('Random Over-Sampling', y_pred_proba_ros_final)
]

for nombre, y_pred_proba in modelos_predicciones:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    ax.plot(fpr, tpr, label=f'{nombre} (AUC = {auc:.3f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio', linewidth=2)
ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
ax.set_title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Curvas Precision vs Recall

# %%
fig, ax = plt.subplots(figsize=(12, 8))

for nombre, y_pred_proba in modelos_predicciones:
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax.plot(recall, precision, label=nombre, linewidth=2)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Curvas Precision-Recall - Comparación de Modelos', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Selecciona el mejor modelo según F1-Score
mejor_modelo_idx = df_resultados['F1-Score'].idxmax()
mejor_modelo_nombre = df_resultados.loc[mejor_modelo_idx, 'Modelo']

print(f"\nMejor modelo según F1-Score: {mejor_modelo_nombre}")

# Obtener coeficientes del mejor modelo
if mejor_modelo_nombre == 'Sin balanceo':
    modelo = best_optuna_model_1
elif mejor_modelo_nombre == 'Class Weight Balanced':
    modelo = best_optuna_model_2
elif mejor_modelo_nombre == 'SMOTE':
    modelo = best_optuna_model_3
elif mejor_modelo_nombre == 'Random Under-Sampling':
    modelo = best_optuna_model_4
else:# mejor_modelo_nombre == 'Random Over-Sampling':
    modelo = best_optuna_model_5

# %%
# importancia de features
coeficientes = pd.DataFrame({
    'Feature': x_train.columns,
    'Coeficiente': modelo.coef_[0]
})
coeficientes['Abs_Coeficiente'] = np.abs(coeficientes['Coeficiente'])
coeficientes = coeficientes.sort_values('Abs_Coeficiente', ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
colores_barras = ['green' if x > 0 else 'red' for x in coeficientes['Coeficiente']]
ax.barh(coeficientes['Feature'], coeficientes['Coeficiente'], color=colores_barras, alpha=0.7)
ax.set_xlabel('Coeficiente', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title(f'Importancia de Features - {mejor_modelo_nombre}', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("\nTop 10 features más importantes:")
print(coeficientes.head(10).to_string(index=False))


# %% [markdown]
# ## Optimización de Umbral de Decisión

# %%
def encontrar_umbral_optimo(y_true, y_pred_proba, metrica='f1'):
    """
    Encuentra el umbral óptimo según la métrica especificada
    
    Parámetros:
    - metrica: 'f1', 'f2' o 'youden'
        - 'f1': Optimiza F1-Score (balance entre precision y recall)
        - 'f2': Optimiza F2-Score (balance entre precision y recall, ponderando mas el recall)
        - 'youden': Optimiza índice de Youden (sensitivity + specificity - 1)
    """
    umbrales = np.linspace(0.01, 0.99, 99)  # Evitamos extremos 0 y 1
    scores = []
    
    for umbral in umbrales:
        y_pred = (y_pred_proba >= umbral).astype(int)
        
        # Verificar que tengamos ambas clases predichas
        if len(np.unique(y_pred)) < 2:
            scores.append(0)
            continue
        
        if metrica == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)

        elif metrica == 'f2':
            score = f2_score(y_true, y_pred, zero_division=0)
            
        elif metrica == 'youden':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1  # Índice de Youden
            
        else:
            raise ValueError("Métrica no reconocida. Usa: 'f1' o 'youden'")
        
        scores.append(score)
    
    idx_optimo = np.argmax(scores)
    return umbrales[idx_optimo], scores[idx_optimo]


def graficar_metricas_por_umbral(y_true, y_pred_proba, nombre_modelo):
    """
    Grafica cómo varían las métricas según el umbral
    """
    umbrales = np.linspace(0.01, 0.99, 99)
    f1_scores = []
    f2_scores = []
    precision_scores = []
    recall_scores = []
    youden_scores = []
    
    for umbral in umbrales:
        y_pred = (y_pred_proba >= umbral).astype(int)
        
        # Verificar que tengamos ambas clases
        if len(np.unique(y_pred)) < 2:
            f1_scores.append(0)
            precision_scores.append(0)
            recall_scores.append(0)
            youden_scores.append(0)
            continue
        
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        f2_scores.append(f2_score(y_true, y_pred, zero_division=0))
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        precision_scores.append(report['1']['precision'])
        recall_scores.append(report['1']['recall'])
        
        # Calcula Youden
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden_scores.append(sensitivity + specificity - 1)


    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plotear métricas
    ax.plot(umbrales, f2_scores, label='F2-Score', linewidth=2.5, color="#DD2E2E")
    ax.plot(umbrales, f1_scores, label='F1-Score', linewidth=2.5, color='#2E86AB')
    ax.plot(umbrales, precision_scores, label='Precision', linewidth=2, 
            alpha=0.7, color='#A23B72', linestyle='--')
    ax.plot(umbrales, recall_scores, label='Recall', linewidth=2, 
            alpha=0.7, color='#F18F01', linestyle='--')
    ax.plot(umbrales, youden_scores, label='Youden Index', linewidth=2.5, 
            color='#06A77D', linestyle='-.')
    
    # Marca umbrales óptimos
    idx_f1_max = np.argmax(f1_scores)
    idx_f2_max = np.argmax(f2_scores)
    idx_youden_max = np.argmax(youden_scores)
    
    
    ax.axvline(umbrales[idx_f1_max], color='#2E86AB', linestyle=':', alpha=0.6, linewidth=2,
               label=f'Óptimo F1 = {umbrales[idx_f1_max]:.2f}')
    ax.axvline(umbrales[idx_f2_max], color='#DD2E2E', linestyle=':', alpha=0.6, linewidth=2,
            label=f'Óptimo F2 = {umbrales[idx_f2_max]:.2f}')
    ax.axvline(umbrales[idx_youden_max], color='#06A77D', linestyle=':', alpha=0.6, linewidth=2,
               label=f'Óptimo Youden = {umbrales[idx_youden_max]:.2f}')
    
    # Marca umbral 0.5 por defecto
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.4, linewidth=1.5,
               label='Umbral por defecto (0.5)')
    
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.4, linewidth=1.5,
               label='0.5')
    
    ax.set_xlabel('Umbral de Decisión', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title(f'Métricas vs Umbral - {nombre_modelo}', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()
    
    return umbrales[idx_f1_max], umbrales[idx_f2_max], umbrales[idx_youden_max]

# %% [markdown]
# ### Análisis de umbrales para cada modelo

# %%
# Diccionarios para almacenar umbrales óptimos
umbrales_optimos_f1 = {}
umbrales_optimos_youden = {}

modelos_info = [
    ('Sin balanceo', y_pred_proba_unbalanced_final, best_optuna_model_1),
    ('Class Weight Balanced', y_pred_proba_balanced_final, best_optuna_model_2),
    ('SMOTE', y_pred_proba_smote_final, best_optuna_model_3),
    ('Random Under-Sampling', y_pred_proba_rus_final, best_optuna_model_4),
    ('Random Over-Sampling', y_pred_proba_ros_final, best_optuna_model_5)
]

print("OPTIMIZACIÓN DE UMBRALES")

for nombre, y_pred_proba, modelo in modelos_info:
    print(f"Modelo: {nombre}")
    
    # Encuentra umbral óptimo para F1
    umbral_f1, score_f1= encontrar_umbral_optimo(y_test, y_pred_proba, 'f1')
    
    # Encuentra umbral óptimo para Youden
    umbral_youden, score_youden= encontrar_umbral_optimo(y_test, y_pred_proba, 'youden')
    
    print(f"\nUMBRALES ÓPTIMOS:")
    print(f"   • Umbral por defecto:    0.500")
    print(f"   • Umbral óptimo F1:      {umbral_f1:.3f}  (F1 = {score_f1:.3f})")
    print(f"   • Umbral óptimo Youden:  {umbral_youden:.3f}  (J = {score_youden:.3f})")
    
    # Guarda los umbrales óptimos
    umbrales_optimos_f1[nombre] = umbral_f1
    umbrales_optimos_youden[nombre] = umbral_youden
    
    # Grafica métricas vs umbral
    graficar_metricas_por_umbral(y_test, y_pred_proba, nombre)
    
    # Compara resultados con umbral 0.5 vs óptimos
    y_pred_05 = (y_pred_proba >= 0.5).astype(int)
    y_pred_opt_f1 = (y_pred_proba >= umbral_f1).astype(int)
    y_pred_opt_youden = (y_pred_proba >= umbral_youden).astype(int)
    
    print(f"\n{'─'*70}")
    print(f"COMPARACIÓN DE MÉTRICAS")
    print(f"{'─'*70}")
    
    print("\n▶ Con umbral por defecto (0.5):")
    print(classification_report(y_test, y_pred_05, target_names=['No llueve', 'Llueve']))
    
    print(f"\n▶ Con umbral óptimo F1 ({umbral_f1:.3f}):")
    print(classification_report(y_test, y_pred_opt_f1, target_names=['No llueve', 'Llueve']))
    
    print(f"\n▶ Con umbral óptimo Youden ({umbral_youden:.3f}):")
    print(classification_report(y_test, y_pred_opt_youden, target_names=['No llueve', 'Llueve']))

# %% [markdown]
#  ### Comparación final con umbrales optimizados

# %%
# Recalcula métricas con umbrales optimizados para F1
resultados_opt_f1 = []

for nombre, y_pred_proba, modelo in modelos_info:
    umbral_opt = umbrales_optimos_f1[nombre]
    y_pred_opt = (y_pred_proba >= umbral_opt).astype(int)
    
    resultados = {
        'Modelo': nombre,
        'Umbral': umbral_opt,
        'Accuracy': accuracy_score(y_test, y_pred_opt),
        'Precision': classification_report(y_test, y_pred_opt, output_dict=True)['1']['precision'],
        'Recall': classification_report(y_test, y_pred_opt, output_dict=True)['1']['recall'],
        'F1-Score': f1_score(y_test, y_pred_opt),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    resultados_opt_f1.append(resultados)

df_resultados_opt_f1 = pd.DataFrame(resultados_opt_f1)

# Recalcula métricas con umbrales optimizados para Youden
resultados_opt_youden = []

for nombre, y_pred_proba, modelo in modelos_info:
    umbral_opt = umbrales_optimos_youden[nombre]
    y_pred_opt = (y_pred_proba >= umbral_opt).astype(int)
    
    resultados = {
        'Modelo': nombre,
        'Umbral': umbral_opt,
        'Accuracy': accuracy_score(y_test, y_pred_opt),
        'Precision': classification_report(y_test, y_pred_opt, output_dict=True)['1']['precision'],
        'Recall': classification_report(y_test, y_pred_opt, output_dict=True)['1']['recall'],
        'F1-Score': f1_score(y_test, y_pred_opt),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    resultados_opt_youden.append(resultados)

df_resultados_opt_youden = pd.DataFrame(resultados_opt_youden)


print("RESULTADOS CON UMBRAL ÓPTIMO F1")
print(df_resultados_opt_f1.to_string(index=False))

print("RESULTADOS CON UMBRAL ÓPTIMO YOUDEN")
print(df_resultados_opt_youden.to_string(index=False))

# %%
# Comparación lado a lado: umbral 0.5 vs F1 vs Youden
df_05 = df_resultados.copy()
df_05['Tipo'] = 'Umbral 0.5'

df_f1 = df_resultados_opt_f1.drop('Umbral', axis=1).copy()
df_f1['Tipo'] = 'Óptimo F1'

df_youden = df_resultados_opt_youden.drop('Umbral', axis=1).copy()
df_youden['Tipo'] = 'Óptimo Youden'

df_combinado = pd.concat([df_05, df_f1, df_youden], ignore_index=True)


# %%
# Visualización comparativa
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

metricas_comparar = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colores = ["#B42D2D", '#2E86AB', '#06A77D']

for idx, metrica in enumerate(metricas_comparar):
    ax = axes[idx // 2, idx % 2]
    
    sns.barplot(data=df_combinado, x='Modelo', y=metrica, hue='Tipo', 
                palette=colores, ax=ax)
    
    ax.set_title(f'{metrica}', fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel(metrica, fontsize=13)
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    plt.setp(ax.get_xticklabels(), ha='right')
    ax.set_ylim(0, 1)
    ax.legend(title='', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

fig.suptitle('Comparación: Umbral 0.5 vs Óptimo F1 vs Óptimo Youden', 
             fontsize=17, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# %%
print("RESUMEN DE UMBRALES ÓPTIMOS POR MODELO")

resumen_umbrales = pd.DataFrame({
    'Modelo': [nombre for nombre, _, _ in modelos_info],
    'Umbral F1': [umbrales_optimos_f1[nombre] for nombre, _, _ in modelos_info],
    'Umbral Youden': [umbrales_optimos_youden[nombre] for nombre, _, _ in modelos_info],
    'Diferencia': [abs(umbrales_optimos_f1[nombre] - umbrales_optimos_youden[nombre]) 
                   for nombre, _, _ in modelos_info]
})

print(resumen_umbrales.to_string(index=False))


# %% [markdown]
# Para optimizar el umbral de clasificación, investigamos el índice J de Youden. Esta métrica, al igual F1-Score en su objetivo de encontrar un umbral óptimo, se enfoca específicamente en maximizar el equilibrio entre la Sensibilidad (Recall) y la Especificidad. El umbral resultante es aquel que maximiza la fórmula $(Sensibilidad + Especificidad - 1)$, identificando así el punto de corte que ofrece el mejor balance. Esto nos permitió lograr un notable Recall (alta detección de positivos) sin sacrificar de manera desproporcionada la Especificidad (la correcta detección de negativos).

# %% [markdown]
# # Comparación de modelos y ajuste fino

# %% [markdown]
# ## Modelo Base

# %%
def modelo_base_rainfall(rainfall):
    aleatorios = np.random.rand(len(rainfall))
    P = 1 / (1 + np.exp(-rainfall))
    predicciones = np.where(P > 0.5, 1, 0)
    return (pd.Series(predicciones), pd.Series(P))


# %%
y_pred_base_rainfall, y_pred_base_rainfall_prob = modelo_base_rainfall(x_test['Rainfall_log'])

resultados_base_rainfall = evaluar_modelo(y_test, y_pred_base_rainfall, y_pred_base_rainfall_prob, 'Modelo Base Rainfall')

print(resultados_base_rainfall)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_base_rainfall, target_names=['No llueve', 'Llueve']))

graficar_matriz_confusion(y_test, y_pred_base_rainfall, 'Modelo Base Rainfall')

# %% [markdown]
# ### SHAP
# Vamos a aplicar al modelo balanceado con random_oversampling, esto es porque fue el que obtuvimos el mejor umbral en la clase `llueve` de la variable target, optimizando el umbral con **Youden**.

# %% [markdown]
# ### Interpretación local

# %%
# comenzamos creando el objeto explainer SHAP

explainer = shap.LinearExplainer(best_optuna_model_5, x_train_ros, feature_names=predictoras)

# %%
shap_values = explainer.shap_values(x_test)

# %% [markdown]
# #### Force Plot (gráfico de fuerza)

# %%
# elegimos una observación cualquiera
# index = 3213 
index = 3219 
# index = 13003


# Force plot para la observación elegida
shap.force_plot(explainer.expected_value, 
                shap_values[index],
                x_test.iloc[index], 
                feature_names=predictoras,
                matplotlib=True, 
                figsize=(18, 4), 
                text_rotation=45)


# información de la observación
print(f"INFORMACIÓN DE LA OBSERVACIÓN {index}")
print(f"Valor real: {'Llueve' if y_test.iloc[index] == 1 else 'No llueve'}")
print(f"Probabilidad predicha: {y_pred_proba_ros_final[index]:.4f}")
print(f"Predicción (umbral 0.5): {'Llueve' if y_pred_ros_final[index] == 1 else 'No llueve'}")

# umbral optimizado Youden
umbral_opt = umbrales_optimos_youden['Random Over-Sampling']
pred_opt = 1 if y_pred_proba_ros_final[index] >= umbral_opt else 0
print(f"Predicción (umbral optimizado con Youden {umbral_opt:.3f}): {'Llueve' if pred_opt == 1 else 'No llueve'}")

# %% [markdown]
# Del gráfico de fuerza notamos que el valor base es la clase mayoritaria "No llueve". En este caso en particular nos encontramos que predice con una probabilidad de 0.7759 que llueve con ambos umbrales, sin optimizar y optimizado con Youden.

# %% [markdown]
# En el gráfico SHAP se observa `f(x) = 1.24`. Este valor está en escala logit (log-odds).
# Probabilidad = 0.7759 es la probabilidad transformada usando la función sigmoide, el valor que se obtiene con y_pred_proba_ros_final[index].
#
# Ambos valores representan lo mismo pero en distintas escalas.

# %% [markdown]
# #### Waterfall plot

# %%
index = 3219 

# Crea el objeto Explanation
explanation = shap.Explanation(values=shap_values[index],
                               base_values=explainer.expected_value,
                               data=x_test.iloc[index].values,
                               feature_names=predictoras)

# Visualiza el waterfall plot
shap.plots.waterfall(explanation)

# Información adicional

print(f"ANÁLISIS DE LA OBSERVACIÓN {index}")

print(f"Valor base (E[f(X)]): {explainer.expected_value:.4f}")
print(f"Predicción final f(x): {explainer.expected_value + shap_values[index].sum():.4f}")
print(f"Probabilidad predicha: {y_pred_proba_ros_final[index]:.4f}")
print(f"Valor real: {'Llueve' if y_test.iloc[index] == 1 else 'No llueve'}")
print(f"Predicción (umbral óptimo {umbrales_optimos_youden['Random Over-Sampling']:.3f}): {'Llueve' if (y_pred_proba_ros_final[index] >= umbrales_optimos_youden['Random Over-Sampling']) else 'No llueve'}")

# %% [markdown]
# La información es la misma, sin embargo, con el gráfico de Waterfall podemos distinguir mejor la influencia de cada una de las variables. 
# De la observación elegida (3219), el gráfico waterfall muestra cómo el modelo construye la predicción.
# Acá notamos más como las variables atmosféricas de presión y humedad son las más determinantes.

# %% [markdown]
# #### Observación de casos particulares.
# A continuación observaremos casos extremos. Los FP y TP que dieron más confianza (probabilidad más alta en la predicción), y el valor más cercano al umbral.

# %%
# Encuentra las predicciones más confiantes (correctas e incorrectas)
y_pred_opt_ros = (y_pred_proba_ros_final >= umbrales_optimos_youden['Random Over-Sampling']).astype(int)

print("CASOS MÁS INTERESANTES PARA ANALIZAR")

# Predicciones más confiantes de "Llueve" que son correctas
tp_probs = y_pred_proba_ros_final[(y_pred_opt_ros == 1) & (y_test == 1)]
if len(tp_probs) > 0:
    idx_tp_max = np.where((y_pred_opt_ros == 1) & (y_test == 1))[0][np.argmax(tp_probs)]
    print(f"\n1. TP más confiante (índice {idx_tp_max}): prob = {y_pred_proba_ros_final[idx_tp_max]:.4f}")

# Predicciones más confiantes de "Llueve" que son incorrectas (Falsos Positivos)
fp_probs = y_pred_proba_ros_final[(y_pred_opt_ros == 1) & (y_test == 0)]
if len(fp_probs) > 0:
    idx_fp_max = np.where((y_pred_opt_ros == 1) & (y_test == 0))[0][np.argmax(fp_probs)]
    print(f"2. FP más confiante (índice {idx_fp_max}): prob = {y_pred_proba_ros_final[idx_fp_max]:.4f}")

# Predicciones cercanas al umbral (casos dudosos)
umbral = umbrales_optimos_youden['Random Over-Sampling']
diff_umbral = np.abs(y_pred_proba_ros_final - umbral)
idx_cercano = np.argmin(diff_umbral)
print(f"3. Predicción más cercana al umbral (índice {idx_cercano}): prob = {y_pred_proba_ros_final[idx_cercano]:.4f}")

# Visualiza estos casos interesantes
casos_especiales = [idx_tp_max if len(tp_probs) > 0 else None,
                   idx_fp_max if len(fp_probs) > 0 else None,
                   idx_cercano]

for i, idx in enumerate(casos_especiales, 1):
    if idx is not None:
        explanation = shap.Explanation(values=shap_values[idx],
                                       base_values=explainer.expected_value,
                                       data=x_test.iloc[idx].values,
                                       feature_names=predictoras)
        
        fig = plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f'Caso Especial {i} - Observación {idx}', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ### Interpretación global
# Ahora continuamos con el escrutinio global de la influencia de las variables en las predicciones del modelo.

# %%
# Crear explanation global con todas las observaciones
explanation_global = shap.Explanation(values=shap_values, 
                                     base_values=explainer.expected_value, 
                                     feature_names=predictoras, 
                                     data=x_test)

print(f"Explanation creada con {explanation_global.shape[0]} observaciones y {explanation_global.shape[1]} features")

# %%
# Bar plot: importancia promedio absoluta de cada feature
print("IMPORTANCIA GLOBAL DE FEATURES (Mean Absolute SHAP)")

shap.plots.bar(explanation_global, max_display=15, show=False)
plt.title('Importancia Global de Features - Random Over-Sampling', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Importancia Global
#
# En este gráfico de barras, podemos observar el **impacto promedio absoluto** de cada feature en las predicciones del modelo Random Over-Sampling.
#
# **Variables más influyentes:**
#
# 1. **Pressure3pm** (+1.14)
# 2. **Humidity3pm** (+0.97)
# 3. **Pressure9am** (+0.86) 
#
# - Las variables meteorológicas de la **tarde** (3pm) son más determinantes que las de la mañana.
# - Las **condiciones atmosféricas** (presión, humedad) son más importantes que la lluvia del día actual.
# - Variables como clima tropical, dirección del viento y diferencia de temperatura tienen un impacto mínimo.
#
# El modelo se basa principalmente en presión y humedad para predecir si lloverá mañana, lo cual tiene sentido meteorológicamente.

# %%
# Beeswarm plot: muestra distribución completa de SHAP values
print("DISTRIBUCIÓN DE SHAP VALUES POR FEATURE")

shap.plots.beeswarm(explanation_global, max_display=15, show=False)
plt.title('Beeswarm Plot - Distribución de Impacto de Features', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# ``Ayuda memoria: si la feature tiene un valor rojo (alto) hacia la izquierda, quiere decir que mientras más aumente el valor de la variable, menos probabilidades hay de que llueva. ``

# %% [markdown]
# El Beeswarm plot revela que Pressure3pm, Humidity3pm y Pressure9am son los features con mayor dispersión de valores SHAP, indicando su rol dominante en las predicciones. Notablemente, Pressure3pm muestra una clara polarización: valores bajos del feature (azul) generan impactos negativos en la predicción, mientras valores altos (rojo) aumentan fuertemente la probabilidad de lluvia, con algunos casos extremos superando +6 de SHAP value.

# %%
# Cohortes según si el modelo acertó o no
y_pred_opt_ros = (y_pred_proba_ros_final >= umbrales_optimos_youden['Random Over-Sampling']).astype(int)
aciertos = (y_pred_opt_ros == y_test.values).astype(int)

aux_aciertos = [
    "Predicción Correcta" if aciertos[i] == 1 else "Predicción Incorrecta"
    for i in range(len(aciertos))
]


print("COHORTES POR ACIERTO DEL MODELO")

shap.plots.bar(explanation_global.cohorts(aux_aciertos).abs.mean(0), show=False)
plt.title('Importancia de Features según Acierto del Modelo', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

print(f"Predicciones Correctas: {aux_aciertos.count('Predicción Correcta')}")
print(f"Predicciones Incorrectas: {aux_aciertos.count('Predicción Incorrecta')}")

# detalle de predicciones correctas y no correctas (tp, tn, fp y fn)
tp = np.sum((y_pred_opt_ros == 1) & (y_test == 1))
tn = np.sum((y_pred_opt_ros == 0) & (y_test == 0))
fp = np.sum((y_pred_opt_ros == 1) & (y_test == 0))
fn = np.sum((y_pred_opt_ros == 0) & (y_test == 1))

print(f"\nDesglose:")
print(f"Verdaderos Positivos: {tp}")
print(f"Verdaderos Negativos: {tn}")
print(f"Falsos Positivos: {fp}")
print(f"Falsos Negativos: {fn}")

# %% [markdown]
# Se observa del gráfico de cohortes por acierto del modelo que Humidity3pm presenta la mayor diferencia entre predicciones correctas e incorrectas. Cuando el modelo predice correctamente, Humidity3pm tiene mayor importancia SHAP (1.02) comparado con predicciones incorrectas (0.76), sugiriendo que niveles de humedad bien caracterizados son clave para el buen desempeño del modelo.
#
# Pressure3pm mantiene consistentemente la mayor importancia SHAP tanto en predicciones correctas (1.15) como incorrectas (1.12), indicando que es el feature más relevante independientemente del desempeño del modelo. En contraste y en relación a lo anteriormente mencionado, Humidity3pm muestra mayor variabilidad (1.02 vs 0.76), siendo más determinante en aciertos que en errores.

# %%
# evaluamos según la media de MinTemp
mediana_variable = x_test['MinTemp'].median()

aux_cohorte = [
    "Alta MinTemp" if x_test['MinTemp'].iloc[i] >= mediana_variable else "Baja MinTemp"
    for i in range(len(x_test))
]

print("COHORTES SEGÚN MEDIANA DE MinTemp")

shap.plots.bar(explanation_global.cohorts(aux_cohorte).abs.mean(0), show=False)
plt.title('Importancia de Features según nivel de MinTemp', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# Estadísticas descriptivas
n_alta = aux_cohorte.count("Alta MinTemp")
n_baja = aux_cohorte.count("Baja MinTemp")
print(f"Casos con MinTemp ≥ mediana: {n_alta}")
print(f"Casos con MinTemp < mediana: {n_baja}")

print(f"\nMediana general de MinTemp: {mediana_variable:.2f}")
print(f"Mediana grupo alto: {x_test.loc[x_test['MinTemp'] >= mediana_variable, 'MinTemp'].mean():.2f}")
print(f"Mediana grupo bajo: {x_test.loc[x_test['MinTemp'] < mediana_variable, 'MinTemp'].mean():.2f}")

# %% [markdown]
# Se eligió MinTemp arbitrariamente para observar el comportamiento en relación al resto de variables
#
# Se **observa** del gráfico de cohortes que MinTemp y Pressure9am tienen una interacción interesante al separar con criterio de mediana:
# Cuando MinTemp es baja (rayado), Pressure9am tiene mayor importancia SHAP (1.01) comparado con cuando MinTemp es alta (sólido), 0.70. Esto sugiere que la presión atmosférica a las 9am es más relevante para la predicción del modelo en días con temperaturas mínimas bajas.

# %%
# resumen de las características globales de SHAP values
print("RESUMEN ESTADÍSTICO DE SHAP VALUES")

shap_df = pd.DataFrame(shap_values, columns=predictoras)
shap_stats = pd.DataFrame({
    'Feature': predictoras,
    'Mean |SHAP|': np.abs(shap_values).mean(axis=0),
    'Std |SHAP|': np.abs(shap_values).std(axis=0),
    'Max |SHAP|': np.abs(shap_values).max(axis=0),
    'Mean SHAP': shap_values.mean(axis=0)
})

shap_stats = shap_stats.sort_values('Mean |SHAP|', ascending=False)
print(shap_stats.head(10).to_string(index=False))

# %% [markdown]
# ## AutoML

# %%
# Concatena las features predictoras con la target para formar un df para pycaret.
variables_pycaret = predictoras.copy()
variables_pycaret.append('RainTomorrow_dummy')


# %%
def pr_auc_score(y_true, y_pred_prob, **kwargs):
    return average_precision_score(y_true, y_pred_prob, )


# %%
clasificacion = classification.setup(
    data = train[variables_pycaret],
    target = 'RainTomorrow_dummy',
    # Deshabilita procesos ya realizados. 
    imputation_type=None,
    normalize=False,
)

classification.set_config('seed', 42)

classification.add_metric('pr_auc', 'PR AUC', pr_auc_score, target='pred_proba', multiclass=False)
classification.add_metric('f2', 'F2-Score', f2_score, multiclass=False)

# %%
# Consideramos unicamente los módelos lineales para comparar con los nuestros.
modelos_lineales = ['lr', 'ridge', 'lda', 'svm']

# Nos quedamos con el mejor priorizando el valor de PR_AUC, ya que será el que tenga un mayor F2-Score con el umbral correcto.
best_model = classification.compare_models(include=modelos_lineales, sort='pr_auc')
print(best_model)

# %% [markdown]
# El mejor módelo resulta ser **Logistic Regression**, que presenta el mejor resultado de *PR-AUC* por lo que maximizará el valor de *F2-Score*, además es el modelo con mayor *AUC*, todo indica que es el mejor fit. No se puede evaluar el valor de *PR_AUC* para los modelos **Ridge Classifier** y **SVM** de pycaret ya que sus implementaciones no exponen las probabilidades predichas, por eso tienen valor 0.0

# %%
classification.evaluate_model(best_model)

# %% [markdown]
# En la sección *Class Report* observamos que el *recall* tiene un marcado desbalance entre clases. Es muy alto (0.94) para la clase mayoritaria (Predice con gran exactitud los días sin lluvia), mientras que solo predice correctamente la mitad de los días con lluvia (0.52). Vamos a optimizar el umbral.

# %%
classification.optimize_threshold(
    best_model,
    optimize="f2"
)

# %% [markdown]
# Analizando el gráfico detectamos que si usamos el umbral óptimo para *F2-score* (0.125) la *Precisión* será muy baja, particularmente < 0.5 . Por lo que vamos a probar con un umbral de 0.2

# %%
predictions = classification.predict_model(best_model, data=test[variables_pycaret], probability_threshold=0.20)

# %%
print(evaluar_modelo(y_test, predictions['prediction_label'], predictions['prediction_score'], "Pycaret LogisticRegression umbral 0.2"))

graficar_matriz_confusion(y_test, predictions['prediction_label'], 'Pycaret LinearDiscriminantAnalysis')

# %% [markdown]
# El módelo con umbral = 0.2 reporta una gran mejora en las métricas que nos interesan. Tiene un *F2-Score = 0.722* y un *Recall = 0.802*, pero manteniendo *Precision = 0.514* .

# %% [markdown]
# # Redes Neuronales

# %%
from sklearn.utils.class_weight import compute_class_weight

pesos = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train), 
    y=y_train
)

# Dict con frecuencia por clase para hacer balanceo en keras
class_weights_dict = dict(enumerate(pesos))

# %%
tf.keras.utils.set_random_seed(42) # Determinismo

X_t, X_val, y_t, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# EarlyStop en caso de que no mejore el pr_auc en 10 épocas seguidas
early_stop = EarlyStopping(
    monitor='val_pr_auc', 
    patience=10, 
    mode='max', 
    restore_best_weights=True,
    verbose=1
)

# Arquitectura simple típica, 2 capas (16 y 8), y un dropout del 0.1 entre capas.
model = Sequential([
    Input(shape=(len(predictoras),)),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='PR', name='pr_auc')])

history = model.fit(
    X_t, y_t,
    validation_data=(X_val, y_val),
    epochs=100, 
    batch_size=256,
    callbacks=[early_stop],
    class_weight=class_weights_dict, # Balanceo de clases
    verbose=1,
)

# %%
#nn_simple = model
#nn_simple.save('nn_simple.keras')
nn_simple = tf.keras.models.load_model('nn_simple.keras')

# %%
y_val_pred_proba = nn_simple.predict(X_val)
y_val_pred = (y_val_pred_proba > 0.5).astype(int)
evaluar_modelo(y_val, y_val_pred, y_val_pred_proba, 'VAL NN Simple 16/8')

# %%
graficar_matriz_confusion(y_val, y_val_pred,'VAL NN Simple 16/8')

# %% [markdown]
# Obtuvimos buenas métricas, *F2-score = 0.733* es un valor alentador teniendo en cuenta que el modelo tiene además *Acuraccy = 0.810* , *Recall = 0.800* y sobretodo un valor de *Precisión = 0.550* que supera comodamente la condición de mayor a 0.5 que nos impusimos. Si bien es un buen fit, vamos a optimizar hiperparametros para tratar de encontrar la mejor arquitectura posible. También vamos a explorar optimización del umbral teniendo en cuenta que nos "Sobra" precisión y posiblemente podamos mejorar un poco más el F2, que es nuestra métrica de referencia.

# %%
graficar_metricas_por_umbral(y_val, y_val_pred_proba, 'NN Simple 16/8')

# %%
y_val_pred = (y_val_pred_proba > 0.38).astype(int)
evaluar_modelo(y_val, y_val_pred, y_val_pred_proba, 'VAL NN Simple 16/8 Umbral 0.38')

# %% [markdown]
# Como con el umbral óptimo de F2 (0.38) el modelo obtiene *Precision = 0.479* que no cumple con nuestro requisito de ser > 0.5. Buscamos el umbral más proximo que si lo cumpla.

# %%
y_val_pred = (y_val_pred_proba > 0.42).astype(int)
evaluar_modelo(y_val, y_val_pred, y_val_pred_proba, 'VAL NN Simple 16/8 Umbral 0.42')

# %%
graficar_matriz_confusion(y_val, y_val_pred, 'VAL NN Simple 16/8 Umbral 0.42')

# %% [markdown]
# Ahora que encontramos el umbral óptimo que cumple con *Precision > 0.5*, vamos a probar el desempeño el test.

# %%
y_pred_proba = nn_simple.predict(x_test)
y_pred = (y_pred_proba > 0.42).astype(int)

evaluar_modelo(y_test, y_pred, y_pred_proba, 'TEST NN Simple 16/8 Umbral 0.42')

# %%
graficar_matriz_confusion(y_test, y_pred, 'TEST NN Simple 16/8 Umbral 0.42')

# %% [markdown]
# ### Optimización de Hiperparámetros con Optuna
#
# Buscamos la arquitectura que maximice el valor de *F2-score* pero manteniendo *Precision > 0.5*. Para esto corremos 100 trials variando la cantidad de capas, cantidad de neuronas y si usa o no Dropout. Monitoreamos el valor de *pr_auc* tanto para el Pruner como para el EarlyStop. Ya que eso nos garantiza un buen valor de *F2_score* en algún umbral.

# %%
import optuna
from tensorflow.keras.callbacks import EarlyStopping
from optuna.integration import TFKerasPruningCallback
from optuna.samplers import TPESampler

def objective(trial):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42) # Determinismo

    X_t, X_v, y_t, y_v = train_test_split(
        x_train, y_train, 
        test_size=0.2, 
        random_state=42, # Fijo para que todos los trials usen los mismos datos
        stratify=y_train
    )

    model = Sequential()
    model.add(Input(shape=(len(predictoras),)))

    dropout_rate = trial.suggest_float(f'dropout_rate', 0.0, 0.2, step=0.1)
    num_layers = trial.suggest_int('num_layers', 2, 3)
    
    for i in range(num_layers):
        num_units = trial.suggest_int(f'n_units_layer_{i}', 8, 128)
        if num_units:
            model.add(Dense(num_units, activation='relu'))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid')) 

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='PR', name='pr_auc')])

    es_interno = EarlyStopping(monitor='val_pr_auc', mode='max', patience=10, verbose=0, restore_best_weights=True)

    history = model.fit(
        X_t, y_t,
        validation_data=(X_v, y_v),
        epochs=50,
        batch_size=256, 
        verbose=0, 
        class_weight=class_weights_dict,
        callbacks=[TFKerasPruningCallback(trial, "val_pr_auc"), es_interno]
    )

    y_pred_proba = model.predict(X_v)
    mejor_f2 = 0.0
    
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_pred_proba > thresh).astype(int)

        precision = precision_score(y_v, y_pred)
        f2 = f2_score(y_v, y_pred)

        if precision > 0.5:
            mejor_f2 = max(f2, mejor_f2)
    
    return mejor_f2

pruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource=50, reduction_factor=3)
sampler = TPESampler(seed=42, n_startup_trials=20)
 
study = optuna.create_study(direction='maximize', pruner=pruner, sampler=sampler)

EJECUTAR = False
####################

# Descomentar para ejecutar
# EJECUTAR = True

####################
if EJECUTAR:
    study.optimize(objective, n_trials=100, n_jobs=1)
    print("Mejores parámetros:", study.best_params)

# %% [markdown]
# Ahora reentrenamos usando la arquitectura óptima encontrada, usamos un conjunto de validación distinto (para verificar que el módelo generaliza) y mas chico (0.10) para extraer la mayor cantidad de información de los datos. Además subimos las épocas a 100 para permitir un entrenamiento más profundo en caso de que la red no se estanque (EarlyStopping sigue activo)

# %%
tf.keras.utils.set_random_seed(42) # Determinismo

X_t, X_val, y_t, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.10,
    random_state=43, # Distinto para no repetir datos
    stratify=y_train
)

# EarlyStop en caso de que no mejore el roc_auc en 10 epócas seguidas
early_stop = callbacks.EarlyStopping(
    monitor='val_pr_auc', 
    patience=10, 
    mode='max', 
    restore_best_weights=True,
    verbose=1
)

# Arquitectura resultante de aplicar hp-tuning con optuna
model = Sequential([
    Input(shape=(len(predictoras),)),
    Dense(121, activation='relu'),
    Dropout(0.1),
    Dense(86, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='PR', name='pr_auc')])

history = model.fit(
    X_t, y_t,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,
    callbacks=[early_stop],
    class_weight=class_weights_dict, # Balanceo de clases
    verbose=1,
)

# %%
#nn_optimizada = model
#nn_optimizada.save('nn_optimizada.keras')
nn_optimizada = tf.keras.models.load_model('nn_optimizada.keras')

# %%
y_val_pred_proba = nn_optimizada.predict(X_val)
y_val_pred = (y_val_pred_proba > 0.5).astype(int)
evaluar_modelo(y_val, y_val_pred, y_val_pred_proba, 'NN Tuned 121/86')

# %%
graficar_matriz_confusion(y_val, y_val_pred, 'NN Tuned 121/86')

# %% [markdown]
# El módelo presenta unas métricas equilibradamente buenas, pero estamos buscando maximizar el *F2-Score* tanto como sea posible siempre que la *Precision* sea > 0.5, vamos a buscar el umbral óptimo para este fin.

# %%
graficar_metricas_por_umbral(y_val, y_val_pred_proba, "NN Tuned 121/86")

# %%
# Evaluamos las métricas con el umbral óptimo F2.
y_val_pred = (y_val_pred_proba > 0.44).astype(int)
evaluar_modelo(y_val, y_val_pred, y_val_pred_proba, 'NN Tuned 121/86 Umbral F2 0.44')

# %%
graficar_matriz_confusion(y_val, y_val_pred, 'NN Tuned 121/86 Umbral F2 0.44')

# %% [markdown]
# El desempeño del modelo con umbral optimizado para F2 mejora considerablemente *F2-Score = 0.786*. Predice casi 90% de los días de lluvia correctamente, que es nuestro principal objetivo. Como *Precision = 0.529* y nuestra condición es que sea > 0.5 vamos a intentar disminuir el umbral para obtener un mayor al actual. *Recall = 0.894*

# %%
y_val_pred = (y_val_pred_proba > 0.40).astype(int)
evaluar_modelo(y_val, y_val_pred, y_val_pred_proba, 'NN Tuned 121/86 Umbral 0.4')

# %%
graficar_matriz_confusion(y_val, y_val_pred, 'NN Tuned 121/86 Umbral 0.4')

# %% [markdown]
# Pudimos mejorar el *Recall* a *0.912* manteniendo *Precision > 0.5*. Logicamente el *F2-Score* también disminuyó, de *0.786* a *0.784* lo cual es insignificante. Consideramos que este umbral es mejor para nuestro objetivo de disminuir al máximo los falsos negativos.

# %% [markdown]
# Evaluamos el desempeño del modelo en test.

# %%
y_pred_proba = nn_optimizada.predict(x_test)
y_pred = (y_pred_proba > 0.40).astype(int)

evaluar_modelo(y_test, y_pred, y_pred_proba, 'TEST NN Optimizada 121/86 Umbral 0.4')

# %%
graficar_matriz_confusion(y_test, y_pred, 'TEST NN Optimizada 121/86 Umbral 0.4')

# %% [markdown]
# El rendimiento del módelo en Test arroja una *precision = 0.49*, lo que indica que nos dejamos llevar por el desempeño sobre los datos de validación y elegimos un umbral poco robusto para nuestro objetivo de que el modelo mantenga una precisión > 0.5 en producción. Vamos a probar el umbral óptimo F2 (0.44), que incrementaba también la precisión, sobre el conjunto de Test.

# %%
y_pred_proba = nn_optimizada.predict(x_test)
y_pred = (y_pred_proba > 0.44).astype(int)

evaluar_modelo(y_test, y_pred, y_pred_proba, 'TEST NN Optimizada 121/86 Umbral 0.44')

# %%
graficar_matriz_confusion(y_test, y_pred, 'TEST NN Optimizada 121/86 Umbral 0.44')

# %% [markdown]
# El rendimiento en test fue marginalmente superior al del módelo simple con umbral optimizado. Asi que nos vamos a quedar con este como módelo de red neuronal.

# %% [markdown]
# ### SHAP para Red Neuronal Optimizada
# Aplicamos SHAP al modelo de red neuronal final (arquitectura 121/86 con umbral 0.44),
# que fue el modelo que obtuvo las mejores métricas basándose en F2-Score.

# %% [markdown]
# ### Interpretación local

# %% [markdown]
# Creamos el explainer SHAP para redes neuronales
# Para redes neuronales usamos DeepExplainer

# Tomamos una muestra del conjunto de entrenamiento como background data
# Es importante convertir a numpy array para evitar errores de formato
#%%
background = X_t[:100].values if hasattr(X_t, 'values') else X_t[:100]

explainer = shap.DeepExplainer(nn_optimizada, background)

# %%
# Calculamos SHAP values para el conjunto de test
x_test_array = x_test[:1000].values if hasattr(x_test, 'values') else x_test[:1000]
shap_values = explainer.shap_values(x_test_array)  # Primeras 1000 obs para eficiencia

# si hsap_values es una lista tomamos el primer elemento
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# %% [markdown]
# #### Análisis de observaciones individuales
# Para redes neuronales, usamos waterfall plots en lugar de force plots

# %%
index = 219

# shap_values puede ser 2D (muestras, features) o 3D (muestras, features, 1)
if len(shap_values.shape) == 3:
    shap_vals_for_plot = shap_values[index, :, 0]
else:
    shap_vals_for_plot = shap_values[index]

# Obtener expected_value
if isinstance(explainer.expected_value, np.ndarray):
    expected_val = explainer.expected_value[0] if len(explainer.expected_value.shape) > 0 else float(explainer.expected_value)
else:
    expected_val = float(explainer.expected_value)

# Crear explanation para visualizacion
explanation = shap.Explanation(values=shap_vals_for_plot,
                               base_values=expected_val,
                               data=x_test.iloc[index].values,
                               feature_names=predictoras)

# waterfall plot
shap.plots.waterfall(explanation)

# Información de la observación
print(f"\nINFORMACIÓN DE LA OBSERVACIÓN {index}")
print(f"Valor real: {'Llueve' if y_test.iloc[index] == 1 else 'No llueve'}")
print(f"Probabilidad predicha: {y_pred_proba[index][0]:.4f}")

# Umbral optimizado F2
umbral_opt = 0.44
pred_opt = 1 if y_pred_proba[index][0] >= umbral_opt else 0
print(f"Predicción (umbral optimizado F2 {umbral_opt}): {'Llueve' if pred_opt == 1 else 'No llueve'}")
print(f"Valor base (E[f(X)]): {expected_val:.4f}")
print(f"Predicción final f(x): {expected_val + shap_vals_for_plot.sum():.4f}")

# %% [markdown]
# Del gráfico waterfall observamos cómo las diferentes variables meteorológicas empujan la predicción 
# hacia "Llueve" o "No llueve" desde el valor base, y poe consiguiente, identificar qué variables son las más determinantes para
# esta predicción específica. La red neuronal captura interacciones más complejas 
# que la regresión logística, lo que puede reflejarse en patrones SHAP diferentes. Se observa también que las mismas variables
# atmosféricas son las dominantes para la red neuroanl:
#
# En la observación 219, el modelo predice correctamente lluvia con 92.57% de probabilidad. Las variables 
# más determinantes son Humidity3pm (+0.36), Pressure3pm (+0.14) y Sunshine (+0.10), todas empujando 
# fuertemente hacia la clase "Llueve". El valor base de 0.336 se incrementa casi 60 puntos porcentuales 
# principalmente por estas tres variables atmosféricas de la tarde, confirmando que las condiciones de 
# humedad alta, presión baja y poco sol son indicadores clave para la predicción de lluvia.

# %% [markdown]
# #### Waterfall plot con una observación (sample) distinta

# %%
index = 500 #elegido arbitrariamente

if len(shap_values.shape) == 3:
    shap_vals_obs = shap_values[index, :, 0]
else:
    shap_vals_obs = shap_values[index]

#objeto Explanation
explanation = shap.Explanation(values=shap_vals_obs,
                               base_values=expected_val,
                               data=x_test.iloc[index].values,
                               feature_names=predictoras)

# Visualiza
shap.plots.waterfall(explanation)

# Información extra
print(f"\nANÁLISIS DE LA OBSERVACIÓN {index}")
print(f"Valor base (E[f(X)]): {expected_val:.4f}")
print(f"Predicción final f(x): {expected_val + shap_vals_obs.sum():.4f}")
print(f"Probabilidad predicha: {y_pred_proba[index][0]:.4f}")
print(f"Valor real: {'Llueve' if y_test.iloc[index] == 1 else 'No llueve'}")
print(f"Predicción (umbral óptimo {umbral_opt}): {'Llueve' if (y_pred_proba[index][0] >= umbral_opt) else 'No llueve'}")

# %% [markdown]
#
# Para la observación 500 (elegida arbitrariamente), el modelo predice lluvia con 53.87% de probabilidad,
# apenas por encima del umbral optimizado (0.44), resultando en una predicción correcta pero con baja
# confianza. Aquí, Pressure3pm domina con +0.26 de contribución, mientras que Pressure9am (-0.13) reduce
# la predicción. Este caso muestra un escenario más ambiguo donde las señales meteorológicas son mixtas:
# presión muy baja en la tarde favorece lluvia, pero otras variables como WindGustSpeed (-0.05) y
# RainySeason (-0.02) empujan levemente hacia "No llueve". La predicción final de 0.539 refleja esta
# incertidumbre, siendo un ejemplo de caso límite donde el modelo está menos seguro.

# %% [markdown]
# #### Observación de casos particulares
# Analizamos casos extremos: FP y TP con mayor confianza, y predicciones cercanas al umbral.

# %%
# Encuentra las predicciones más interesantes
y_pred_opt_nn = (y_pred_proba[:1000] >= umbral_opt).astype(int).flatten()
y_test_subset = y_test.iloc[:1000].values

print("CASOS MÁS INTERESANTES PARA ANALIZAR")

# Verdaderos Positivos más confiantes
tp_mask = (y_pred_opt_nn == 1) & (y_test_subset == 1)
if tp_mask.sum() > 0:
    tp_probs = y_pred_proba[:1000][tp_mask]
    idx_tp_max = np.where(tp_mask)[0][np.argmax(tp_probs)]
    print(f"\n1. TP más confiante (índice {idx_tp_max}): prob = {y_pred_proba[idx_tp_max][0]:.4f}")
else:
    idx_tp_max = None

# Falsos Positivos más confiantes
fp_mask = (y_pred_opt_nn == 1) & (y_test_subset == 0)
if fp_mask.sum() > 0:
    fp_probs = y_pred_proba[:1000][fp_mask]
    idx_fp_max = np.where(fp_mask)[0][np.argmax(fp_probs)]
    print(f"2. FP más confiante (índice {idx_fp_max}): prob = {y_pred_proba[idx_fp_max][0]:.4f}")
else:
    idx_fp_max = None

# Predicción más cercana al umbral
diff_umbral = np.abs(y_pred_proba[:1000].flatten() - umbral_opt)
idx_cercano = np.argmin(diff_umbral)
print(f"3. Predicción más cercana al umbral (índice {idx_cercano}): prob = {y_pred_proba[idx_cercano][0]:.4f}")

# Visualizar casos especiales
casos_especiales = [idx_tp_max, idx_fp_max, idx_cercano]
titulos = ['TP más confiante', 'FP más confiante', 'Caso cercano al umbral']

for i, (idx, titulo) in enumerate(zip(casos_especiales, titulos), 1):
    if idx is not None:
        # Extraer SHAP values correctamente
        if len(shap_values.shape) == 3:
            shap_vals_obs = shap_values[idx, :, 0]
        else:
            shap_vals_obs = shap_values[idx]
            
        explanation = shap.Explanation(values=shap_vals_obs,
                                       base_values=expected_val,
                                       data=x_test.iloc[idx].values,
                                       feature_names=predictoras)
        
        fig = plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f'{titulo} - Observación {idx}', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
# %% [markdown]
# ### Análisis de Casos Extremos
#
# **1. TP más confiante (Observación 253 - prob = 0.9969):**
# Este es un verdadero positivo con máxima confianza. Las variables clave son Humidity3pm (+0.28) y 
# Pressure3pm (+0.25), ambas con valores extremos (2.094 y -2.199 respectivamente) que señalan 
# condiciones atmosféricas muy favorables para lluvia. La combinación de humedad muy alta y presión 
# muy baja resulta en una predicción casi perfecta (99.69%). Sunshine negativo (+0.06) y WindGustSpeed 
# alto (+0.05) refuerzan la predicción. El modelo muestra certeza alta cuando las variables 
# atmosféricas presentan valores extremos que también son coherentes.
#
# **2. FP más confiante (Observación 197 - prob = 0.9950):**
# Este falso positivo revela una debilidad del modelo. Humidity3pm extremadamente alta (2.287) aporta 
# +0.33, dominando completamente la predicción. Pressure3pm (-1.117, +0.14), Sunshine bajo (+0.08) y 
# Rainfall_log (+0.06) refuerzan la señal de lluvia. Sin embargo, el modelo **se equivoca** porque 
# confía excesivamente en la humedad sin considerar suficientemente otros factores. Interesantemente, 
# Pressure9am (-0.06) y WindGustSpeed (-0.02) aportan señales negativas débiles que fueron ignoradas. 
# Este caso ilustra que humedad extrema puede generar falsos positivos.
#
# **3. Caso cercano al umbral (Observación 884 - prob = 0.4413):**
# Esta predicción está a solo 0.001 del umbral optimizado (0.44), representando máxima incertidumbre. 
# Las variables empujan en direcciones opuestas: WindGustSpeed (+0.08), Sunshine bajo (+0.07) y TempDiff 
# (+0.05) favorecen lluvia, mientras que WindSpeed9am (-0.03), Humidity3pm (-0.03), Pressure9am (-0.03) 
# y WindDir3pm_sin (-0.03) empujan hacia "No llueve". Con una probabilidad de 44.13%, el modelo está 
# prácticamente en el límite de la decisión, reflejando que las señales meteorológicas son 
# contradictorias y ambiguas para este día específico. Ninguna aporta considerablemnte a la predicción.
# %% [markdown]
# ### Interpretación global
# Analizamos la influencia global de las variables en las predicciones de la red neuronal.

# %%
# Crear explanation global
# Ajustar shap_values para que sea 2D
if len(shap_values.shape) == 3:
    shap_values_2d = shap_values[:, :, 0]
else:
    shap_values_2d = shap_values

# Asegurar que expected_value es escalar
if isinstance(explainer.expected_value, np.ndarray):
    expected_val_scalar = explainer.expected_value[0] if len(explainer.expected_value.shape) > 0 else float(explainer.expected_value)
else:
    expected_val_scalar = float(explainer.expected_value)

# Para explanation_global, base_values debe ser un array del mismo tamaño que las muestras
base_values_array = np.full(shap_values_2d.shape[0], expected_val_scalar)

explanation_global = shap.Explanation(values=shap_values_2d, 
                                     base_values=base_values_array, 
                                     feature_names=predictoras, 
                                     data=x_test.iloc[:1000].values)

print(f"Explanation creada con {explanation_global.shape[0]} observaciones y {explanation_global.shape[1]} features")

# %%
# Bar plot: importancia promedio absoluta de cada feature
print("IMPORTANCIA GLOBAL DE FEATURES (Mean Absolute SHAP)")

shap.plots.bar(explanation_global, max_display=15, show=False)
plt.title('Importancia Global de Features - Red Neuronal Optimizada', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Importancia Global en la Red Neuronal
#
# El gráfico de barras muestra el **impacto promedio absoluto** de cada feature en las predicciones del modelo.
#
# **Variables más influyentes:**
#
# 1. **Pressure3pm** (+0.15) - La presión atmosférica de la tarde es el predictor más importante
# 2. **Humidity3pm** (+0.13) - La humedad de la tarde es el segundo predictor más relevante  
# 3. **Pressure9am** (+0.10) - La presión de la mañana completa el top 3
#
# **Observaciones clave:**
#
# - Las **variables de la tarde** (3pm) dominan claramente, con Pressure3pm y Humidity3pm liderando el ranking
# - Las **condiciones atmosféricas** (presión y humedad) son mucho más determinantes que otras variables meteorológicas
# - **WindGustSpeed** (+0.05) y **Sunshine** (+0.05) tienen importancia moderada
# - Variables como temperatura (MinTemp, TempDiff), nubosidad (Cloud3pm) y lluvia en el día (Rainfall_log) tienen impacto menor (+0.02-0.03)
# - Variables de clima regional (ClimateTropical) y direcciones de viento codificadas tienen impacto mínimo (+0.01-0.02)
#
# Este ranking es consistente con el modelo de regresión logística analizado previamente, confirmando que 
# la red neuronal también identifica las condiciones de presión y humedad como los factores más críticos 
# para predecir lluvia al día siguiente, seguidos por WindGustSpeed y Sunshine.
# %%
# Beeswarm plot: distribución completa de SHAP values
print("DISTRIBUCIÓN DE SHAP VALUES POR FEATURE")

shap.plots.beeswarm(explanation_global, max_display=15, show=False)
plt.title('Beeswarm Plot - Red Neuronal Optimizada', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# El Beeswarm plot revela cómo cada valor de feature impacta en las predicciones. 
# Las redes neuronales pueden capturar relaciones no lineales, por lo que podríamos captar 
# patrones más complejos que en la regresión logística.
#
# **Patrones destacados:**
#
# - **Pressure3pm**: Muestra una clara relación lineal inversa. Valores bajos (azul, a la izquierda) 
# generan impactos positivos fuertes (hasta +0.75 SHAP, y un dato atípico en casi +1), mientras valores altos (rojo, a la derecha) 
# producen impactos negativos (hasta casi -0.50 SHAP). Esto continúa reforzando lo visto hasta el momento: 
# **la presión baja en la tarde es el indicador más fuerte de lluvia.**
#
# - **Humidity3pm**: Presenta una relación directa marcada. Valores altos generan impactos 
# positivos significativos (hasta +0.60 aproximadamente), mientras valores bajos (azul) reducen la probabilidad de lluvia. 
# La dispersión es amplia, indicando alta variabilidad en su influencia.
#
# - **Pressure9am**: Llamativamente contrario a Pressure3pm. Valores bajos (azul) se concentran 
# en la izquierda (impacto negativo), valores altos (rojo) a la derecha (impacto positivo).
#
# - **WindGustSpeed** y **Sunshine**: Muestran patrones más complejos con dispersión bidireccional, 
# sugiriendo que la red neuronal captura interacciones no lineales. Valores altos de WindGustSpeed 
# (rojo) tienden a impacto positivo, mientras que poco Sunshine (azul) también favorece lluvia.
#
# - **Variables secundarias** (MinTemp, Cloud3pm, TempDiff): Presentan distribuciones más simétricas 
# alrededor de cero, con impactos modestos pero contribuciones consistentes en ambas direcciones.
#
# El patrón general confirma que presión y humedad dominan las predicciones, pero la red neuronal 
# captura relaciones más matizadas que un modelo lineal para variables secundarias.

# %%
# Cohortes según acierto del modelo
y_pred_opt_nn_full = (y_pred_proba >= umbral_opt).astype(int).flatten()
aciertos = (y_pred_opt_nn_full[:1000] == y_test.iloc[:1000].values).astype(int)

aux_aciertos = [
    "Predicción Correcta" if aciertos[i] == 1 else "Predicción Incorrecta"
    for i in range(len(aciertos))
]

print("COHORTES POR ACIERTO DEL MODELO")

shap.plots.bar(explanation_global.cohorts(aux_aciertos).abs.mean(0), show=False)
plt.title('Importancia de Features según Acierto - Red Neuronal', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

print(f"Predicciones Correctas: {aux_aciertos.count('Predicción Correcta')}")
print(f"Predicciones Incorrectas: {aux_aciertos.count('Predicción Incorrecta')}")

# Desglose de predicciones
tp = np.sum((y_pred_opt_nn_full[:1000] == 1) & (y_test.iloc[:1000] == 1))
tn = np.sum((y_pred_opt_nn_full[:1000] == 0) & (y_test.iloc[:1000] == 0))
fp = np.sum((y_pred_opt_nn_full[:1000] == 1) & (y_test.iloc[:1000] == 0))
fn = np.sum((y_pred_opt_nn_full[:1000] == 0) & (y_test.iloc[:1000] == 1))

print(f"\nDesglose:")
print(f"Verdaderos Positivos: {tp}")
print(f"Verdaderos Negativos: {tn}")
print(f"Falsos Positivos: {fp}")
print(f"Falsos Negativos: {fn}")

# %% [markdown]
# El análisis por cohortes revela qué features son más importantes cuando el modelo acierta vs cuando falla.
# Esto puede ayudar a identificar debilidades del modelo o condiciones donde no generaliza bien.
#
# **Hallazgos del análisis por cohortes (800 correctas vs 200 incorrectas):**
#
# - **Pressure3pm**: Muestra diferencia moderada entre aciertos (0.14) y errores (0.16). En predicciones 
# incorrectas, esta variable tiene **mayor importancia**, sugiriendo que el modelo puede "sobre-confiar" en 
# presiones extremas que no siempre correlacionan con lluvia.
#
# - **Humidity3pm**: Mantiene importancia casi idéntica en ambas cohortes (~0.13), indicando que es un 
# predictor robusto tanto en aciertos como en errores. No es una fuente principal de confusión del modelo.
#
# - **Pressure9am**: Importancia mayor en errores (0.11) vs aciertos (0.09), sugiriendo que 
# la presión de las 9am puede generar señales confusas cuando no está alineada con las condiciones de la tarde.
#
# - **WindGustSpeed**: Notable diferencia: 0.05 en aciertos vs **0.08 en errores**. Los vientos fuertes 
# parecen confundir al modelo. Posiblemente sea porque ocurren independientemente de si llueve o no.
#
# - **Variables restantes**: La mayoría (Sunshine, MinTemp, Cloud3pm, TempDiff) mantienen importancias 
# similares en ambas cohortes, sugiriendo contribuciones estables.
#
# **Diagnóstico**: Los 170 falsos positivos (85% de los errores) representan un porcentaje bastante 
# contundente de sobre-predicción de lluvia. Esto es consistente con la estrategia de optimización: 
# al priorizar **recall** de la clase positiva mediante el **F2-Score** y usar un umbral ligeramente bajo (0.44),
# estamos intencionalmente capturando más 
# lluvias de las que realmente ocurren. Este trade-off es aceptable dado nuestro objetivo de **minimizar 
# falsos negativos** (solo 30), sacrificando precisión en favor de no perder días lluviosos. Las variables 
# Pressure3pm, Pressure9am y WindGustSpeed tienen mayor peso en estos errores, sugiriendo que condiciones 
# atmosféricas ambiguas generan confianza excesiva del modelo incluso cuando no llueve finalmente.
# %%
# Cohortes según variable meteorológica clave
variable_analizar = 'Humidity3pm' 
mediana_variable = x_test[variable_analizar].median()

aux_cohorte = [
    f"Alta {variable_analizar}" if x_test[variable_analizar].iloc[i] >= mediana_variable 
    else f"Baja {variable_analizar}"
    for i in range(len(x_test[:1000]))
]

print(f"COHORTES SEGÚN MEDIANA DE {variable_analizar}")

shap.plots.bar(explanation_global.cohorts(aux_cohorte).abs.mean(0), show=False)
plt.title(f'Importancia de Features según nivel de {variable_analizar}', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# Estadísticas
n_alta = aux_cohorte.count(f"Alta {variable_analizar}")
n_baja = aux_cohorte.count(f"Baja {variable_analizar}")
print(f"Casos con {variable_analizar} ≥ mediana: {n_alta}")
print(f"Casos con {variable_analizar} < mediana: {n_baja}")

print(f"\nMediana general de {variable_analizar}: {mediana_variable:.2f}")

# %% [markdown]
# El análisis por cohortes de variables meteorológicas revela cómo cambia la importancia de las features
# bajo diferentes condiciones atmosféricas. Esto puede mostrar interacciones que la red neuronal captura.
#
# Se eligió **Humidity3pm arbitrariamente** para observar el comportamiento del modelo bajo diferentes 
# niveles de humedad. La mediana de Humidity3pm (estandarizada) es 0.02, dividiendo el conjunto en dos 
# grupos casi balanceados: 501 casos con alta humedad vs 499 con baja humedad.
#
# **Observaciones clave:**
#
# - **Pressure3pm**: Mantiene importancia idéntica (0.15) en ambas cohortes, confirmando que es un 
# predictor robusto independientemente del nivel de humedad. La presión atmosférica actúa como señal 
# consistente.
#
# - **Humidity3pm**: Llama la atención que tiene **mayor importancia cuando es baja** (0.14) que cuando es 
# alta (0.12). Esto sugiere que la red neuronal captura que humedades bajas son más "informativas" para 
# descartar lluvia, mientras que humedades altas, aunque correlacionan con lluvia, pueden ser menos 
# discriminativas por sí solas.
#
# - **WindGustSpeed**: Notable diferencia: 0.06 con alta humedad vs 0.04 con baja humedad. Los vientos 
# fuertes tienen mayor relevancia cuando ya hay humedad alta, esto lo podemos relacionar con lo observado en gráficos anteriores que
# se había concluido que no parecían influir mucho, y en este caso se observa que sí parece intervenir, 
# pero cuando la humedad también es alta. 
#
# - **MinTemp**: Mayor importancia con alta humedad (0.04) vs baja (0.02), sugiriendo que la temperatura 
# mínima interactúa con la humedad.
#
# Este análisis revela que la red neuronal **captura interacciones no lineales** entre variables: la 
# importancia relativa de features cambia dependiendo del contexto meteorológico (nivel de humedad).
# %%
# Resumen estadístico de SHAP values
print("RESUMEN ESTADÍSTICO DE SHAP VALUES - RED NEURONAL")

shap_df = pd.DataFrame(shap_values_2d, columns=predictoras)
shap_stats = pd.DataFrame({
    'Feature': predictoras,
    'Mean |SHAP|': np.abs(shap_values_2d).mean(axis=0),
    'Std |SHAP|': np.abs(shap_values_2d).std(axis=0),
    'Max |SHAP|': np.abs(shap_values_2d).max(axis=0),
    'Mean SHAP': shap_values_2d.mean(axis=0)
})

shap_stats = shap_stats.sort_values('Mean |SHAP|', ascending=False)
print(shap_stats.head(10).to_string(index=False))

# %% [markdown]
# ### Tabla Comparativa: Regresión Logística vs Red Neuronal
#
# Para facilitar la comparación, mostramos nuevamente los resultados obtenidos previamente con regresión logística 
# junto a los de la red neuronal optimizada:
#
# **Importancia de Features (Top 5):**
#
# | Ranking | Regresión Logística | Mean \|SHAP\| RL | Red Neuronal | Mean \|SHAP\| NN |
# |---------|---------------------|------------------|--------------|------------------|
# | 1 | Pressure3pm | 1.107 | Pressure3pm | 0.147 |
# | 2 | Humidity3pm | 0.960 | Humidity3pm | 0.132 |
# | 3 | Pressure9am | 0.823 | Pressure9am | 0.096 |
# | 4 | WindGustSpeed | 0.516 | WindGustSpeed | 0.054 |
# | 5 | Sunshine | 0.328 | Sunshine | 0.049 |
#
# Los valores absolutos de SHAP difieren significativamente entre modelos debido a diferencias 
# en las escalas de trabajo (logit vs activaciones de red neuronal), pero el **ranking relativo** se 
# mantiene consistente, confirmando que ambos modelos identifican las mismas variables como críticas 
# para la predicción de lluvia.
# %% [markdown]

# ### Comparación con Regresión Logística
# 
# **Resumen:**
# 
# **1. Ranking de importancia - Consistencia notable:**
# 
# Ambos modelos identifican las mismas tres variables como más importantes:
# - **Pressure3pm**: #1 en ambos (NN: 0.147 | RL: 1.107)
# - **Humidity3pm**: #2 en ambos (NN: 0.132 | RL: 0.960)
# - **Pressure9am**: #3 en ambos (NN: 0.096 | RL: 0.823)
# 
# El top 5 también es idéntico: las tres anteriores más WindGustSpeed (#4) y Sunshine (#5), confirmando 
# que las condiciones atmosféricas de presión y humedad son los predictores fundamentales independientemente 
# de la arquitectura del modelo.
#
# **2. Magnitud de impacto - Diferencias de escala:**
# 
# Los valores SHAP de la regresión logística son **7-10 veces mayores** que los de la red neuronal (ej: 
# Pressure3pm 1.11 vs 0.15). Esto es esperado: la RL trabaja en escala logit y tiene una relación lineal 
# directa entre features y predicción, mientras que la NN distribuye el impacto a través de múltiples capas 
# y activaciones no lineales. Las **proporciones relativas** se mantienen, lo importante es el ranking.
#
# **3. Patrones de interacción - Capacidad de la NN:**
# 
# El beeswarm plot de la NN muestra relaciones más complejas que la RL, especialmente en variables secundarias 
# como WindGustSpeed y MinTemp, donde la NN captura efectos bidireccionales más matizados. Sin embargo, 
# para las variables principales (Pressure3pm, Humidity3pm), ambos modelos muestran patrones similares, 
# sugiriendo que estas relaciones son fundamentalmente lineales o monotónicas.
#
# **4. Aciertos vs errores - Patrones similares:**
# 
# Tanto en NN como en RL, las predicciones incorrectas muestran mayor dependencia de Pressure3pm y 
# WindGustSpeed, indicando que condiciones atmosféricas ambiguas confunden a ambos modelos de manera similar.
#
# **Conclusión:** La red neuronal replica el conocimiento de la regresión logística sobre qué variables 
# son importantes, pero añade capacidad de capturar interacciones no lineales sutiles. Para este problema 
# de predicción de lluvia, las relaciones fundamentales parecen ser suficientemente lineales, explicando 
# por qué ambos modelos convergen al mismo ranking de importancia.

# %% [markdown]
# ## Comparación de modelos
resultados_nn_optimizada = evaluar_modelo(y_test, y_pred, y_pred_proba, 'TEST NN Optimizada 121/86 Umbral 0.44')
mejor_lr = df_resultados_opt_youden[df_resultados_opt_youden['Modelo'] == 'Random Over-Sampling'].iloc[0].to_dict()
comparativa_final = pd.DataFrame([mejor_lr, resultados_nn_optimizada])

plt.figure(figsize=(10, 6))
sns.barplot(data=comparativa_final, x='Modelo', y='F2-Score', palette='muted')
plt.title('Comparación Final F2-Score: LR vs NN')
plt.show()

# %% [markdown]
# # MLOPS

# %% [markdown]
# Implementamos un pipeline para replicar las transformaciones aplicadas durante el feature engineering, comparando que el resultado obtenido sea el mismo. Generamos funciones de transformación propias compatibles con Sklearn.pipeline en el archivo *custom_transformers.py*

# %%
# Generamos un df x_train_clean a partir del df original imputado (antes del feature engineering)

x_train_clean = train_imputed.drop('RainTomorrow', axis=1)

# %%
cols_log = ['Rainfall', 'Evaporation']
cols_wind = ['WindDir9am', 'WindDir3pm', 'WindGustDir']

cols_to_scale = [
    'Rainfall_log', 'Evaporation_log', 'Sunshine', 
    'MinTemp', 'TempDiff', 
    'Humidity9am', 'Humidity3pm', 
    'Pressure9am', 'Pressure3pm', 
    'WindSpeed9am', 'WindSpeed3pm', 'WindGustSpeed'
]

# Agrega dinámicamente las variables de viento encodeadas sin/cos
for col in cols_wind:
    cols_to_scale.extend([f'{col}_sin', f'{col}_cos'])

# Pipeline de Feature Engineering
feature_eng_pipeline = Pipeline([
    ('climate', ClimateTransformer()),
    ('season', RainySeasonTransformer()),
    ('log', LogTransformer(cols=cols_log)),
    ('temp_diff', TempDiffTransformer()),
    ('cloud_scale', CloudScalerTransformer()),
    ('wind_cyc', WindCyclicalTransformer(cols=cols_wind))
])

# Pipeline FE + escalado
pipeline = Pipeline([
    ('feature_engineering', feature_eng_pipeline),
    ('scale', ColumnTransformer([
        ('scaler', StandardScaler(), cols_to_scale),
        ('pass', 'passthrough', ['Cloud9am', 'Cloud3pm', 'RainySeason', 'ClimateArid', 'ClimateTropical'])
    ], verbose_feature_names_out=False))
])


# %%
# Fitear el pipeline con los datos de train (para los scalers)
pipeline.fit(x_train_clean)

x_train_clean = pipeline.transform(x_train_clean)

# %%
nombres_cols = pipeline.named_steps['scale'].get_feature_names_out()

x_train_clean = pd.DataFrame(x_train_clean, columns=nombres_cols)

pd.DataFrame(x_train_clean).describe() # df generado aplicando el pipeline

# %%
x_train.describe() # df generado durante el feature engineering / estandarización

# %% [markdown]
# Podemos observar que los dataframes obtenidos son equivalentes y por lo tanto concluir que el pipeline de transformación es correcto.

# %% [markdown]
# ### Pipeline de imputación

# %% [markdown]
# Generamos un imputer para sklearn replicando la estrategía utilizada durante el entrenamiento, pero eliminando lós métodos que dependen de datos del mismo día, para poder correr en producción.

# %%
from custom_imputer import CustomImputer

# %%
df_clean = pd.read_csv('weatherAUS.csv')
train_clean, _ = train_test_split(df_clean, test_size=0.2, random_state=1)
x_train_clean = train_clean.drop('RainTomorrow', axis=1)

# %%
# Redefine el pipeline para sacar a ClimateTransformer ya que
# necesitamos la variable Climate generada para hacer la imputación.

feature_eng_pipeline = Pipeline([
    ('season', RainySeasonTransformer()),
    ('log', LogTransformer(cols=cols_log)),
    ('temp_diff', TempDiffTransformer()),
    ('cloud_scale', CloudScalerTransformer()),
    ('wind_cyc', WindCyclicalTransformer(cols=cols_wind))
])

fe_scaling_pipeline = Pipeline([
    ('feature_engineering', feature_eng_pipeline),
    ('scale', ColumnTransformer([
        ('scaler', StandardScaler(), cols_to_scale),
        ('pass', 'passthrough', ['Cloud9am', 'Cloud3pm', 'RainySeason', 'ClimateArid', 'ClimateTropical'])
    ], verbose_feature_names_out=False))
])

pipeline_completo = Pipeline([
    ('climate', ClimateTransformer()),
    ('imputer', CustomImputer()),
    ('fe-scaler', fe_scaling_pipeline)
    ])

# %%
pipeline_completo.fit(x_train_clean)

# %%
# Dumpeamos el pipeline completo para usarlo en docker
joblib.dump(pipeline_completo, 'pipeline.joblib')

# Dumpeamos el orden exacto de columnas que Keras espera recibir
joblib.dump(x_test.columns.to_list(), 'features_order.joblib')
