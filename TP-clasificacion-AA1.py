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

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

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
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
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
# ## Modelo 1: Regresión Logística (sin balanceo)

# %%
lr_base = LogisticRegression(max_iter=1000, random_state=42)
lr_base.fit(x_train, y_train)

y_pred_base = lr_base.predict(x_test)
y_pred_proba_base = lr_base.predict_proba(x_test)[:, 1]

resultados_base = evaluar_modelo(y_test, y_pred_base, y_pred_proba_base, 'Sin balanceo')

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_base, target_names=['No llueve', 'Llueve']))

graficar_matriz_confusion(y_test, y_pred_base, 'Sin balanceo')

# %% [markdown]
# ## Modelo 2: Regresión Logística con balanceo de clase

# %%
lr_balanced = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_balanced.fit(x_train, y_train)

y_pred_balanced = lr_balanced.predict(x_test)
y_pred_proba_balanced = lr_balanced.predict_proba(x_test)[:, 1]

resultados_balanced = evaluar_modelo(y_test, y_pred_balanced, y_pred_proba_balanced, 'Class Weight Balanced')

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_balanced, target_names=['No llueve', 'Llueve']))

graficar_matriz_confusion(y_test, y_pred_balanced, 'Class Weight Balanced')

# %% [markdown]
# ## Modelo 3: Regresión Lógistica con SMOTE

# %%
# Aplicar SMOTE al conjunto de entrenamiento
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

print(f"Distribución después de SMOTE:")
print(pd.Series(y_train_smote).value_counts())

lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(x_train_smote, y_train_smote)

y_pred_smote = lr_smote.predict(x_test)
y_pred_proba_smote = lr_smote.predict_proba(x_test)[:, 1]

resultados_smote = evaluar_modelo(y_test, y_pred_smote, y_pred_proba_smote, 'SMOTE')

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_smote, target_names=['No llueve', 'Llueve']))

graficar_matriz_confusion(y_test, y_pred_smote, 'SMOTE')

# %% [markdown]
# ## Modelo 4: Regresión Logística con Random Under-Samplig

# %%
# Aplicar Random Under-Sampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(x_train, y_train)

print(f"Distribución después de Under-Sampling:")
print(pd.Series(y_train_rus).value_counts())

lr_rus = LogisticRegression(max_iter=1000, random_state=42)
lr_rus.fit(X_train_rus, y_train_rus)

y_pred_rus = lr_rus.predict(x_test)
y_pred_proba_rus = lr_rus.predict_proba(x_test)[:, 1]

resultados_rus = evaluar_modelo(y_test, y_pred_rus, y_pred_proba_rus, 'Random Under-Sampling')

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_rus, target_names=['No llueve', 'Llueve']))

graficar_matriz_confusion(y_test, y_pred_rus, 'Random Under-Sampling')


# %% [markdown]
# ## Modelo 5: Regresión Logística con Random Over-Sampling

# %%
# Aplicar Random Over-Sampling
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)

print(f"Distribución después de Over-Sampling:")
print(pd.Series(y_train_ros).value_counts())

lr_ros = LogisticRegression(max_iter=1000, random_state=42)
lr_ros.fit(X_train_ros, y_train_ros)

y_pred_ros = lr_ros.predict(x_test)
y_pred_proba_ros = lr_ros.predict_proba(x_test)[:, 1]

resultados_ros = evaluar_modelo(y_test, y_pred_ros, y_pred_proba_ros, 'Random Over-Sampling')

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_ros, target_names=['No llueve', 'Llueve']))

graficar_matriz_confusion(y_test, y_pred_ros, 'Random Over-Sampling')

# %%
# Crear DataFrame con todos los resultados
df_resultados = pd.DataFrame([
    resultados_base,
    resultados_balanced,
    resultados_smote,
    resultados_rus,
    resultados_ros
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
    ('Sin balanceo', y_pred_proba_base),
    ('Class Weight Balanced', y_pred_proba_balanced),
    ('SMOTE', y_pred_proba_smote),
    ('Random Under-Sampling', y_pred_proba_rus),
    ('Random Over-Sampling', y_pred_proba_ros)
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
    modelo = lr_base
elif mejor_modelo_nombre == 'Class Weight Balanced':
    modelo = lr_balanced
elif mejor_modelo_nombre == 'SMOTE':
    modelo = lr_smote
elif mejor_modelo_nombre == 'Random Under-Sampling':
    modelo = lr_rus
else:# mejor_modelo_nombre == 'Random Over-Sampling':
    modelo = lr_ros

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
    - metrica: 'f1' o 'youden'
        - 'f1': Optimiza F1-Score (balance entre precision y recall)
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
            
        elif metrica == 'youden':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1  # Índice de Youden
            
        else:
            raise ValueError("Métrica no reconocida. Usa: 'f1' o 'youden'")
        
        scores.append(score)
    
    idx_optimo = np.argmax(scores)
    return umbrales[idx_optimo], scores[idx_optimo], umbrales, scores


def graficar_metricas_por_umbral(y_true, y_pred_proba, nombre_modelo):
    """
    Grafica cómo varían las métricas según el umbral
    """
    umbrales = np.linspace(0.01, 0.99, 99)
    f1_scores = []
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
    ax.plot(umbrales, f1_scores, label='F1-Score', linewidth=2.5, color='#2E86AB')
    ax.plot(umbrales, precision_scores, label='Precision', linewidth=2, 
            alpha=0.7, color='#A23B72', linestyle='--')
    ax.plot(umbrales, recall_scores, label='Recall', linewidth=2, 
            alpha=0.7, color='#F18F01', linestyle='--')
    ax.plot(umbrales, youden_scores, label='Youden Index', linewidth=2.5, 
            color='#06A77D', linestyle='-.')
    
    # Marca umbrales óptimos
    idx_f1_max = np.argmax(f1_scores)
    idx_youden_max = np.argmax(youden_scores)
    
    ax.axvline(umbrales[idx_f1_max], color='#2E86AB', linestyle=':', alpha=0.6, linewidth=2,
               label=f'Óptimo F1 = {umbrales[idx_f1_max]:.2f}')
    ax.axvline(umbrales[idx_youden_max], color='#06A77D', linestyle=':', alpha=0.6, linewidth=2,
               label=f'Óptimo Youden = {umbrales[idx_youden_max]:.2f}')
    
    # Marca umbral 0.5 por defecto
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.4, linewidth=1.5,
               label='Umbral por defecto (0.5)')
    
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
    
    return umbrales[idx_f1_max], umbrales[idx_youden_max]


# %% [markdown]
# ### Análisis de umbrales para cada modelo

# %%
# Diccionarios para almacenar umbrales óptimos
umbrales_optimos_f1 = {}
umbrales_optimos_youden = {}

modelos_info = [
    ('Sin balanceo', y_pred_proba_base, lr_base),
    ('Class Weight Balanced', y_pred_proba_balanced, lr_balanced),
    ('SMOTE', y_pred_proba_smote, lr_smote),
    ('Random Under-Sampling', y_pred_proba_rus, lr_rus),
    ('Random Over-Sampling', y_pred_proba_ros, lr_ros)
]

print("=" * 80)
print("OPTIMIZACIÓN DE UMBRALES")
print("=" * 80)

for nombre, y_pred_proba, modelo in modelos_info:
    print(f"\n{'='*70}")
    print(f"Modelo: {nombre}")
    print('='*70)
    
    # Encuentra umbral óptimo para F1
    umbral_f1, score_f1, _, _ = encontrar_umbral_optimo(y_test, y_pred_proba, 'f1')
    
    # Encuentra umbral óptimo para Youden
    umbral_youden, score_youden, _, _ = encontrar_umbral_optimo(y_test, y_pred_proba, 'youden')
    
    print(f"\n📊 UMBRALES ÓPTIMOS:")
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

print("\n" + "=" * 100)
print("RESULTADOS CON UMBRAL ÓPTIMO F1")
print("=" * 100)
print(df_resultados_opt_f1.to_string(index=False))

print("\n" + "=" * 100)
print("RESULTADOS CON UMBRAL ÓPTIMO YOUDEN")
print("=" * 100)
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
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

metricas_comparar = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colores = ['#E8E8E8', '#2E86AB', '#06A77D']

for idx, metrica in enumerate(metricas_comparar):
    ax = axes[idx // 2, idx % 2]
    
    sns.barplot(data=df_combinado, x='Modelo', y=metrica, hue='Tipo', 
                palette=colores, ax=ax)
    
    ax.set_title(f'{metrica}', fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel(metrica, fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(title='', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

fig.suptitle('Comparación: Umbral 0.5 vs Óptimo F1 vs Óptimo Youden', 
             fontsize=17, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# %%
print("\n" + "=" * 80)
print("RESUMEN DE UMBRALES ÓPTIMOS POR MODELO")
print("=" * 80)

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
