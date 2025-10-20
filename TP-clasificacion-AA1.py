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
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve, f1_score)

from scipy.stats import chi2_contingency

# %%
# Carga el dataset en un dataframe
df = pd.read_csv('weatherAUS.csv')

# Revisa si hay filas duplicadas
df.duplicated().sum() # 0 filas duplicadas
df.describe(include='all')

# %%
df["RainTomorrow"].value_counts(normalize=True).round(2)

# %% [markdown]
# El dataset está desbalanceado. 78% clase 0 u 22% clase 1.

# %%
# distribución de 'RainTomorrow'
plt.figure(figsize=(8, 6))
sns.countplot(x='RainTomorrow', stat='proportion', data=df, )
plt.title('Distribución de RainTomorrow', fontsize=16)
plt.xlabel('¿Lloverá Mañana?')
plt.ylabel('Proporción')
plt.tight_layout()
plt.grid()
plt.show()

# %%
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0}).astype('Int8')
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0}).astype('Int8')

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
# Drop de filas con NaN en la feature objetivo. Justificación: preferimos evitar imputar la variable objetivo y correr el riesgo de introducir ruido en el dataset, porque la cantidad de datos que perdemos es relativamente baja (menos del 2.25% del dataset).
df = df.dropna(subset=['RainTomorrow'])
# %%
# Drop de filas con mas de la mitad de features con valor nulo
df = df[df.isna().sum(axis=1) <= 11]

# %% [markdown]
# # Análisis de las variables Cloud9am y Cloud3pm

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
# # Imputación de valores faltantes
# Vamos a imputar los valores faltantes por cercanía, por fecha y por región. Primero calculamos las distancias entre las locaciones, y a cada una le asignamos un lugar más cercano. 
# Luego agrupamos por fecha y se iteran entre todas las fechas. Si la locación más cercana es del mismo clima, se obtiene el valor faltante del registro hecho en el mismo día.
# De no poderse imputar algunos valores (porque la locación cercana no es del mismo clima o porque ambas tienen faltantes en el mismo día y variable) se imputan con la mediana en los valores numéricos y con la moda en los valores categóricos.

# %%
from scipy.spatial.distance import cdist

def imputar_por_proximidad(df, coords_path='australian_locations.csv', location_koppen=None):
    """
    Imputa valores faltantes usando dos fases:
    IMPUTACIÓN 1: Por ubicación más cercana del mismo clima y mismo día
    IMPUTACIÓN 2: Por mediana (numéricas) o moda (categóricas) para los restantes
    
    Parameters:
    -----------
    df : DataFrame
        Dataset con datos meteorológicos
    coords_path : str
        Ruta al archivo CSV con coordenadas (columns: location, lat, lon)
    location_koppen : dict
        Diccionario con clasificación climática por ubicación
        
    Returns:
    --------
    DataFrame con valores imputados y reporte de imputación
    """
    
    # Cargar coordenadas
    coords = pd.read_csv(coords_path)
    
    # Añadir Climate al dataframe de coordenadas 
    if location_koppen:
        coords['Climate'] = coords['location'].map(location_koppen)
    
    # Calcular matriz de distancias UNA SOLA VEZ
    locations = coords['location'].values
    coords_array = coords[['lat', 'lon']].values
    dist_matrix = cdist(coords_array, coords_array, metric='euclidean')
    
    # Para cada ubicación, encontrar las más cercanas del mismo clima
    nearest_by_climate = {}
    
    for i, loc in enumerate(locations):
        climate = coords.iloc[i]['Climate']
        # Filtrar ubicaciones del mismo clima
        same_climate_mask = coords['Climate'] == climate
        same_climate_indices = coords[same_climate_mask].index.tolist()
        
        # Obtener distancias a ubicaciones del mismo clima (excluyendo la misma)
        distances = [(idx, dist_matrix[i, idx]) for idx in same_climate_indices if idx != i]
        # Ordenar por distancia
        distances.sort(key=lambda x: x[1])
        
        # Guardar lista de ubicaciones ordenadas por proximidad
        nearest_by_climate[loc] = [locations[idx] for idx, _ in distances]
    
    print(f"  → Matriz de distancias calculada para {len(locations)} ubicaciones")
    
    # Añadir Climate al dataframe si no existe
    df_imputed = df.copy()
    if 'Climate' not in df_imputed.columns and location_koppen:
        df_imputed['Climate'] = df_imputed['Location'].map(location_koppen)
    
    # Identificar columnas a imputar
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['lat', 'lon']]
    
    categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    binary_cols = ['RainToday'] # PENSAR SI IMPUTAMOS RainTomorrow !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    all_cols_to_impute = numeric_cols + categorical_cols + binary_cols
    
    # Reporte de imputación
    reporte = {col: {'imputacion1': 0, 'imputacion2': 0, 'total_nulos': df_imputed[col].isna().sum()} 
               for col in all_cols_to_impute if col in df_imputed.columns}
    
    print("\nIMPUTACIÓN 1: por proximidad geográfica")
    
    # Agrupar por Date
    grouped = df_imputed.groupby('Date')
    
    # IMPUTACIÓN 1: Imputación por proximidad
    for col in all_cols_to_impute:
        if col not in df_imputed.columns:
            continue
            
        total_missing = df_imputed[col].isna().sum()
        if total_missing == 0:
            continue
        
        print(f"\nProcesando {col}: {total_missing} valores faltantes")
        imputados_imputacion1 = 0
        
        # Procesar por fecha para reducir búsquedas
        for date, group in grouped:
            # Identificar filas con valores faltantes en este grupo
            missing_in_group = group[group[col].isna()]
            
            if len(missing_in_group) == 0:
                continue
            
            # Crear un diccionario de valores disponibles por ubicación en esta fecha
            available_values = group[group[col].notna()].set_index('Location')[col].to_dict()
            
            # Imputar cada fila faltante
            for idx, row in missing_in_group.iterrows():
                location = row['Location']
                
                # Obtener lista de ubicaciones cercanas (ya ordenadas por proximidad)
                if location not in nearest_by_climate:
                    continue
                
                nearest_locations = nearest_by_climate[location]
                
                # Buscar la primera ubicación cercana que tenga el valor disponible
                for nearest_loc in nearest_locations:
                    if nearest_loc in available_values:
                        df_imputed.loc[idx, col] = available_values[nearest_loc]
                        imputados_imputacion1 += 1
                        break
        
        reporte[col]['imputacion1'] = imputados_imputacion1
        print(f"  → Imputados en IMPUTACIÓN 1: {imputados_imputacion1} ({imputados_imputacion1/total_missing*100:.1f}%)")
    
    print("\nIMPUTACIÓN 2: con mediana/moda")
    
    for col in all_cols_to_impute:
        if col not in df_imputed.columns:
            continue
            
        missing_mask = df_imputed[col].isna()
        num_missing = missing_mask.sum()
        
        if num_missing == 0:
            continue
        
        if col in categorical_cols + binary_cols:
            # Moda para categóricas
            moda = df_imputed[col].mode()
            if len(moda) > 0:
                df_imputed.loc[missing_mask, col] = moda[0]
                reporte[col]['imputacion2'] = num_missing
                print(f"{col}: {num_missing} imputados con moda '{moda[0]}'")
        else:
            # Mediana para numéricas
            mediana = df_imputed[col].median()
            df_imputed.loc[missing_mask, col] = mediana
            reporte[col]['imputacion2'] = num_missing
            print(f"{col}: {num_missing} imputados con mediana {mediana:.2f}")
    
    # Crear DataFrame de reporte
    reporte_df = pd.DataFrame(reporte).T
    reporte_df['total_imputados'] = reporte_df['imputacion1'] + reporte_df['imputacion2']
    reporte_df['porcentaje_imputacion1'] = (reporte_df['imputacion1'] / reporte_df['total_nulos'] * 100).round(2)
    reporte_df['porcentaje_imputacion2'] = (reporte_df['imputacion2'] / reporte_df['total_nulos'] * 100).round(2)
    
    print("\nREPORTE DE IMPUTACIÓN:")
    print(reporte_df[reporte_df['total_nulos'] > 0].to_string())
    
    return df_imputed, reporte_df



# %%
# df_imputado, reporte = imputar_por_proximidad(
#     df, 
#     coords_path='australian_locations.csv',
#     location_koppen=location_koppen
# )

# # si se quiere guardar el resultado (primera ejecución)
# df_imputado.to_csv('weather_data_imputed.csv', index=False)

# si ya fue generado anteriormente, descomentar la línea inferior y comentar el resto para simplemente cargar y evitar reprocesamiento
df_imputado = pd.read_csv('weather_data_imputed.csv') 

# %%
faltantes_df = pd.DataFrame({
    'NaN': df_imputado.isna().sum(),
    '%': (df_imputado.isna().sum() / len(df_imputado) * 100).round(2)
}).sort_values('NaN', ascending=False)
faltantes_df


# %%
def validar_imputacion(df_original, df_imputado, columnas=None, figsize=(15, 10)):
    """
    Valida que la imputación no haya alterado significativamente las distribuciones.
    
    Parameters:
    -----------
    df_original : DataFrame
        Dataset original con valores faltantes
    df_imputado : DataFrame
        Dataset después de la imputación
    columnas : list, optional
        Lista de columnas a validar. Si None, valida todas las numéricas.
    figsize : tuple
        Tamaño de las figuras
        
    Returns:
    --------
    DataFrame con estadísticas comparativas
    """
    
    # Seleccionar columnas numéricas si no se especifican
    if columnas is None:
        columnas = df_original.select_dtypes(include=[np.number]).columns.tolist()
        columnas = [col for col in columnas if col not in ['lat', 'lon']]
    
    # Dataframe para resultados estadísticos
    resultados = []
 
    print("Comparación de distribuciones antes y después de la imputación:")
    
    for col in columnas:
        if col not in df_original.columns or col not in df_imputado.columns:
            continue
        
        # Datos originales (sin NaN)
        original_sin_nan = df_original[col].dropna()
        # Datos imputados completos
        imputado_completo = df_imputado[col].dropna()
        # Solo los valores que fueron imputados
        mask_imputados = df_original[col].isna() & df_imputado[col].notna()
        valores_imputados = df_imputado.loc[mask_imputados, col]
        
        if len(valores_imputados) == 0:
            continue
        
        # Calcular estadísticas
        stats_dict = {
            'variable': col,
            'n_imputados': len(valores_imputados),
            'pct_imputados': len(valores_imputados) / len(df_imputado) * 100,
            
            # Medidas de tendencia central
            'media_original': original_sin_nan.mean(),
            'media_imputado': imputado_completo.mean(),
            'diff_media': imputado_completo.mean() - original_sin_nan.mean(),
            
            'mediana_original': original_sin_nan.median(),
            'mediana_imputado': imputado_completo.median(),
            'diff_mediana': imputado_completo.median() - original_sin_nan.median(),
            
            # Medidas de dispersión
            'std_original': original_sin_nan.std(),
            'std_imputado': imputado_completo.std(),
            'diff_std': imputado_completo.std() - original_sin_nan.std(),
        }
        
        resultados.append(stats_dict)
    
    df_resultados = pd.DataFrame(resultados)
    
    # Mostrar resumen
    print("\nRESUMEN ESTADÍSTICO")
    print(df_resultados[['variable', 'n_imputados', 'pct_imputados', 
                         'diff_media', 'diff_mediana', 'diff_std']].to_string(index=False))
    return df_resultados


def visualizar_comparacion_distribuciones(df_original, df_imputado, columnas=None, 
                                         max_cols=None, figsize=(18, 12)):
    """
    Visualiza comparación de distribuciones antes y después de imputación usando KDE plots.
    
    Parameters:
    -----------
    df_original : DataFrame
        Dataset original con valores faltantes
    df_imputado : DataFrame
        Dataset después de la imputación
    columnas : list, optional
        Lista de columnas a visualizar. Si None, selecciona todas las numéricas.
    max_cols : int, optional
        Número máximo de columnas a visualizar. Si None, visualiza todas.
    """
    
    # Seleccionar columnas si no se especifican
    if columnas is None:
        columnas_numericas = df_original.select_dtypes(include=[np.number]).columns.tolist()
        columnas_numericas = [col for col in columnas_numericas if col not in ['lat', 'lon']]
        
        # Ordenar por cantidad de NaN
        nans_por_col = [(col, df_original[col].isna().sum()) for col in columnas_numericas]
        nans_por_col.sort(key=lambda x: x[1], reverse=True)
        
        if max_cols is not None:
            columnas = [col for col, _ in nans_por_col[:max_cols]]
        else:
            columnas = [col for col, _ in nans_por_col]
    
    n_cols = len(columnas)
    n_rows = (n_cols + 2) // 3
    
    # Ajustar figsize si hay muchas variables
    if n_rows > 4:
        figsize = (18, n_rows * 3)
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    # Variable para guardar handles y labels de la leyenda (solo una vez)
    legend_handles = None
    legend_labels = None
    
    for idx, col in enumerate(columnas):
        ax = axes[idx]
        
        # Datos
        original_sin_nan = df_original[col].dropna()
        imputado_completo = df_imputado[col].dropna()
        mask_imputados = df_original[col].isna() & df_imputado[col].notna()
        valores_imputados = df_imputado.loc[mask_imputados, col]
        
        # Preparar datos para KDE plots
        df_plot_original = pd.DataFrame({
            'valor': original_sin_nan,
            'tipo': 'Original'
        })
        
        df_plot_completo = pd.DataFrame({
            'valor': imputado_completo,
            'tipo': 'Con imputación'
        })
        
        df_plot_imputados = pd.DataFrame({
            'valor': valores_imputados,
            'tipo': 'Solo imputados'
        })
        
        # Combinar para el plot
        df_combined = pd.concat([df_plot_original, df_plot_completo, df_plot_imputados])
        
        # KDE plots con seaborn
        sns.kdeplot(
            data=df_combined, 
            x='valor', 
            hue='tipo',
            hue_order=['Original', 'Con imputación', 'Solo imputados'],
            palette={'Original': '#3498db', 'Con imputación': '#2ecc71', 'Solo imputados': '#e74c3c'},
            ax=ax,
            common_norm=False,
            fill=True,
            alpha=0.3,
            linewidth=2.5,
            legend=False  # Desactivar leyenda individual
        )
        
        # Líneas verticales para las medias
        ax.axvline(original_sin_nan.mean(), color='#3498db', linestyle='--', 
                   linewidth=2, alpha=0.8)
        ax.axvline(imputado_completo.mean(), color='#2ecc71', linestyle='--', 
                   linewidth=2, alpha=0.8)
        
        # Capturar handles y labels solo del primer plot para la leyenda general
        if idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        
        ax.set_title(f'{col}\n({len(valores_imputados)} imputados, '
                    f'{len(valores_imputados)/len(df_imputado)*100:.1f}%)',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Valor', fontsize=9)
        ax.set_ylabel('Densidad', fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
    
    # ocultar ejes vacíos
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    # leyenda general única en la parte superior de la figura
    fig.legend(
        ['Original (datos pre-existentes)', 
         'Con imputación (dataset completo)', 
         'Solo imputados (valores agregados)',
         'Media Original',
         'Media Con imputación'],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        title='Leyenda de Distribuciones',
        title_fontsize=12
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Dejar espacio para la leyenda
    plt.show()




def analisis_completo_imputacion(df_original, df_imputado, columnas_clave=None):
    """
    Ejecuta análisis completo de validación de imputación.
    """
    
    # 1. Estadísticas comparativas
    df_stats = validar_imputacion(df_original, df_imputado, columnas_clave)
    
    # 2. Visualizaciones
    print("COMPARACIÓN GRÁFICA de distribuciones antes y después de la imputación:")
    visualizar_comparacion_distribuciones(df_original, df_imputado, columnas_clave)
    
    return df_stats


# %%
df_stats = analisis_completo_imputacion(df, df_imputado)

# %% [markdown]
# El análisis gráfico muestra que el proceso de imputación mantiene las distribuciones originales del conjunto de datos.

# %% [markdown]
# No obstante, en etapas posteriores del trabajo identificamos una limitación metodológica: la imputación debería realizarse 
# *después* de la separación train-test para prevenir data leakage. El procedimiento correcto consistiría en imputar 
# los valores faltantes del conjunto de test basándose exclusivamente en las estadísticas y datos del conjunto de entrenamiento.
# Por lo tanto dejamos para etapas posteriores la re-implementación de la imputación siguiendo este enfoque más riguroso.


# %%
df = df_imputado
# %% [markdown]
# ### Split Train/Test

# %%
# Separa el 80% para train y 20% para test
train, test= train_test_split(df, test_size=0.2, random_state=1) # stratify para evitar el problema de desbalanceo

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

# %%
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(4, 4, figsize=(20, 18))

axes = axes.flatten()

for i, col in enumerate(variables_numericas):
    sns.boxplot(data=df, x='RainTomorrow', y=col, ax=axes[i])

    if col in ['Rainfall', 'Evaporation']:
        axes[i].set_yscale('log')  # Usar escala logarítmica para estas variables

    axes[i].set_title(f'Distribución de {col}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('', fontsize=12)
    axes[i].set_ylabel('', fontsize=12)


plt.suptitle('Distribución según si Llueve al Día Siguiente (RainTomorrow)', fontsize=24, y=1.02, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
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


# %% [markdown]
# Análisis y tratamiento de Date

# %%
train.sample(3)

# %%
train['Date'] = pd.to_datetime(train['Date'])
train['Month'] = train['Date'].dt.month
climate_monthly_rain = train.groupby(['Climate', 'Month'])['RainTomorrow'].mean().reset_index()


# Usamos catplot para crear fácilmente subplots para cada categoría de 'Climate'
g = sns.catplot(
    data=climate_monthly_rain,
    x='Month',
    y='RainTomorrow',
    col='Climate',  # Crea una columna de gráficos para cada valor de 'Climate'
    kind='bar',     # Especifica que queremos un gráfico de barras
    palette='viridis',
    height=5,       # Altura de cada gráfico
    aspect=1.5      # Relación ancho/alto
)

# --- Paso 4: Mejorar la legibilidad del gráfico ---
# Títulos y etiquetas
g.fig.suptitle('Probabilidad de Lluvia Mensual por Zona Climática', y=1.03, fontsize=16)
g.set_axis_labels('Mes del Año', 'Probabilidad de Lluvia')
g.set_titles("Zona: {col_name}")

# Cambiar las etiquetas del eje X a nombres de meses
month_labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
g.set_xticklabels(month_labels)

plt.show()

# %% [markdown]
# Para capturar la naturaleza cíclica de la variable, y observando que no existen distribuciones bimodales, transformamos la variable `Date` a seno y coseno.

# %%
# nos aseguramos de que date sea datetime (es posible que ya lo hayamos hecho antes, sacar si es así) REVISAR
train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])

# Basamos la transformación en el día del año
train['DayOfYear'] = train['Date'].dt.dayofyear
test['DayOfYear'] = test['Date'].dt.dayofyear

#Convertir a radianes y calcular seno/coseno
day_of_year_radians_train = (train['DayOfYear'] / 365.25) * 2 * np.pi
train['DayOfYear_sin'] = np.sin(day_of_year_radians_train)
train['DayOfYear_cos'] = np.cos(day_of_year_radians_train)
day_of_year_radians_test = (test['DayOfYear'] / 365.25) * 2 * np.pi
test['DayOfYear_sin'] = np.sin(day_of_year_radians_test)
test['DayOfYear_cos'] = np.cos(day_of_year_radians_test)

print("Resultado de la transformación en el DataFrame de entrenamiento:")
print(train[['Date', 'DayOfYear', 'DayOfYear_sin', 'DayOfYear_cos']].head())
print("\nResultado de la transformación en el DataFrame de prueba:")
print(test[['Date', 'DayOfYear', 'DayOfYear_sin', 'DayOfYear_cos']].head())


# %% [markdown]
# # Análisis de la Influencia de la Dirección del Viento

# %% [markdown]
# ###  Análisis Gráfico de `WindGustDir`

# %%
train_eda = train.copy()
train_eda['RainTomorrow'] = train_eda['RainTomorrow'].astype('category')

# cada proporcion de lluvia para cada dirección del viento
wind_rain_proportion = train_eda.groupby('WindGustDir')['RainTomorrow'].value_counts(normalize=True).unstack()

# ordena 
wind_rain_proportion = wind_rain_proportion.sort_values(by=1.0, ascending=False)

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
tabla_de_contingencia = pd.crosstab(train['WindGustDir'], train['RainTomorrow'])

# prueba de Chi-Cuadrado
chi2, p_value, dof, expected = chi2_contingency(tabla_de_contingencia)

print("--- Prueba de Chi-Cuadrado para WindGustDir y RainTomorrow ---")
print(f"Estadístico Chi2: {chi2:.4f}")
print(f"P-valor: {p_value}")

# p-value
if p_value < 0.05:
    print("\nConclusión: El p-valor es menor que 0.05. Se rechaza la hipótesis nula.")
    print("Existe una asociación estadísticamente significativa entre la dirección del viento y si lloverá mañana.")
else:
    print("\nConclusión: El p-valor es mayor que 0.05. No se puede rechazar la hipótesis nula.")
    print("No hay evidencia de una asociación estadísticamente significativa entre la dirección del viento y si lloverá mañana.")

# %% [markdown]
# Hipótesis: el viento es relevante en decirnos si llueve mañana siempre en cuando esté en consonancia con la ubicación de la costa de la ciudad. Recordemos que la gran mayoría de locaciones que poseemos en el dataset son costeras o muy cercanas a una costa. Por lo tanto, si la costa está al oeste, un viento oeste podría implicar más posibilidad de lluvia, y viceversa. El reciente análisis nos dice que los vientos del este son menos relacionados con lluvia pero tendemos a creer que esto sucede porque simplemente el presente dataset posee menos locaciones con costas al este.
#
# Para obtener la información sobre relevancia de la dirección del viento según costa más cercana, vamos a generar una variable relacionada con las costa.

# %%
direccion_costa = {
    # --- Costa al Este (E) ---
    'Brisbane': 'E', 'Canberra': 'E', 'CoffsHarbour': 'E', 'GoldCoast': 'E',
    'MountGinini': 'E', 'Newcastle': 'E', 'NorahHead': 'E', 'NorfolkIsland': 'E',
    'Penrith': 'E', 'Richmond': 'E', 'Sydney': 'E', 'SydneyAirport': 'E',
    'Tuggeranong': 'E', 'Williamtown': 'E', 'Wollongong': 'E', 'BadgerysCreek': 'E',

    # --- Costa al Oeste (W) ---
    'Perth': 'W', 'PerthAirport': 'W', 'PearceRAAF': 'W', 'Witchcliffe': 'W',

    # --- Costa al Norte (N) ---
    'Cairns': 'N', 'Darwin': 'N', 'Katherine': 'N', 'Launceston': 'N', 'Townsville': 'N',

    # --- Costa al Sur (S) ---
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
# Mapeo de las 16 direcciones a ángulos 
wind_dir_map = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

wind_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

# aplicamos a test y train
for df in [train, test]:
    for col in wind_cols:
        angles = df[col].map(wind_dir_map)
        
        # radianes
        radians = np.deg2rad(angles)
        
        # Calcular seno y coseno 
        df[f'{col}_sin'] = np.sin(radians)
        df[f'{col}_cos'] = np.cos(radians)
        
print("Variables cíclicas creadas.")


# %% [markdown]
# Vamos a generar una variable `IsOnShoreWind` que distinga si el viento viene del mar o de la masa continental.
# Como tenemos tres variables de dirección del viento, vamos a distinguir cual es más importante para predecir RainTomorrow. Suponemos que `WindGustDir` y `WindDir3pm` son más relevantes que `WindDir9am`

# %%
# Comparamos WindGustDir, WindDir9am y WindDir3pm para ver cuál tiene la relación más fuerte con la lluvia cuando se convierte a una variable onshore/offshore. para evitar data leakageeste análisis se realiza solo sobre el conjunto de train.

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
    # Imputar NaNs para el análisis (ESTO QUITARLO CUANDO SE HAGA EL IMPUTADO FINAL)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    moda = train_analysis[col].mode()[0]
    train_analysis[col] = train_analysis[col].fillna(moda)
    
    train_analysis[f'IsOnshore_{col}'] = train_analysis.apply(es_viento_marino, axis=1, wind_col_name=col)

# comparacion visual y estadistica
best_wind_var = ''
max_chi2 = -1

for col in wind_cols:
    onshore_col = f'IsOnshore_{col}'


    plt.figure(figsize=(6, 4))
    sns.barplot(data=train_analysis, x=onshore_col, y='RainTomorrow', palette='coolwarm',hue=onshore_col)
    plt.title(f'Probabilidad de Lluvia vs. {onshore_col}')
    plt.ylabel('Proporción de Lluvia')
    plt.xticks([0, 1], ['Offshore', 'Onshore'])
    plt.show()
    
    # Prueba Chi-Cuadrado
    contingency_table = pd.crosstab(train_analysis[onshore_col], train_analysis['RainTomorrow'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"{onshore_col}")
    print(f"Chi-Cuadrado: {chi2:.2f}")
    
    if chi2 > max_chi2:
        max_chi2 = chi2
        best_wind_var = col

print(f"\nConclusión")
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

# drop de columnas originales y auxiliares
train.drop(columns=['CoastDirection'] + wind_cols, inplace=True)
test.drop(columns=['CoastDirection'] + wind_cols, inplace=True)

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
# ### 1.2 Codificación de Variables Categóricas (Dummies). ==AUXILIAR TAMBIÉN==

# %%
categorical_cols = X_train.select_dtypes(include=['object']).columns

# One-Hot Encoding
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
# # Pipeline Comparativo: Estrategias de Manejo de Desbalanceo en Regresión Logística

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, classification_report, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Función para Entrenar y Evaluar Modelos

# %%
def train_and_evaluate(X_train, y_train, X_test, y_test, method_name, 
                       model=None, apply_threshold_tuning=True):
    """
    Entrena un modelo de regresión logística y calcula métricas de evaluación.
    
    Parámetros:
    -----------
    X_train, y_train : Features y target de entrenamiento
    X_test, y_test : Features y target de prueba
    method_name : str, nombre del método para identificación
    model : modelo preconfigurado (opcional)
    apply_threshold_tuning : bool, si se debe buscar el umbral óptimo
    
    Retorna:
    --------
    dict con métricas, predicciones y modelo entrenado
    """
    
    # Entrenar modelo
    if model is None:
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)
    
    # Búsqueda de umbral óptimo (maximizando F1-Score)
    best_threshold = 0.5
    y_pred_optimal = y_pred_default.copy()
    
    if apply_threshold_tuning:
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * recall[:-1] * precision[:-1] / (recall[:-1] + precision[:-1] + 1e-10)
        best_threshold = thresholds[np.argmax(f1_scores)]
        y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
    
    # Calcular métricas con umbral por defecto (0.5)
    metrics_default = {
        'method': method_name,
        'threshold': 0.5,
        'accuracy': accuracy_score(y_test, y_pred_default),
        'precision': precision_score(y_test, y_pred_default),
        'recall': recall_score(y_test, y_pred_default),
        'f1': f1_score(y_test, y_pred_default),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Calcular métricas con umbral óptimo
    metrics_optimal = {
        'method': method_name,
        'threshold': best_threshold,
        'accuracy': accuracy_score(y_test, y_pred_optimal),
        'precision': precision_score(y_test, y_pred_optimal),
        'recall': recall_score(y_test, y_pred_optimal),
        'f1': f1_score(y_test, y_pred_optimal),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return {
        'model': model,
        'metrics_default': metrics_default,
        'metrics_optimal': metrics_optimal,
        'y_pred_proba': y_pred_proba,
        'y_pred_default': y_pred_default,
        'y_pred_optimal': y_pred_optimal,
        'best_threshold': best_threshold
    }

# %% [markdown]
# ## 2. Ejecución de las 4 Estrategias

# %%
# Diccionario para almacenar resultados
results = {}

print("="*80)
print("ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
print("="*80)

# %% [markdown]
# ### 2.1 Método 1: Sin Corrección (Baseline)

# %%
print("\n1. Modelo Base (Sin corrección de desbalanceo)")
print("-" * 60)

results['baseline'] = train_and_evaluate(
    X_train, y_train, X_test, y_test, 
    method_name='Sin Corrección'
)

print(f"✓ Entrenamiento completado")
print(f"  Umbral óptimo encontrado: {results['baseline']['best_threshold']:.4f}")

# %% [markdown]
# ### 2.2 Método 2: Oversampling (SMOTE)

# %%
print("\n2. Modelo con Oversampling (SMOTE)")
print("-" * 60)

# Aplicar SMOTE solo en el conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"  Distribución original: {dict(pd.Series(y_train).value_counts())}")
print(f"  Distribución después de SMOTE: {dict(pd.Series(y_train_smote).value_counts())}")

results['oversampling'] = train_and_evaluate(
    X_train_smote, y_train_smote, X_test, y_test,
    method_name='Oversampling (SMOTE)'
)

print(f"✓ Entrenamiento completado")
print(f"  Umbral óptimo encontrado: {results['oversampling']['best_threshold']:.4f}")

# %% [markdown]
# ### 2.3 Método 3: Undersampling

# %%
print("\n3. Modelo con Undersampling")
print("-" * 60)

# Aplicar Random Undersampling solo en el conjunto de entrenamiento
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print(f"  Distribución original: {dict(pd.Series(y_train).value_counts())}")
print(f"  Distribución después de undersampling: {dict(pd.Series(y_train_rus).value_counts())}")

results['undersampling'] = train_and_evaluate(
    X_train_rus, y_train_rus, X_test, y_test,
    method_name='Undersampling'
)

print(f"✓ Entrenamiento completado")
print(f"  Umbral óptimo encontrado: {results['undersampling']['best_threshold']:.4f}")

# %% [markdown]
# ### 2.4 Método 4: Class Weights

# %%
print("\n4. Modelo con Class Weights")
print("-" * 60)

# Crear modelo con class_weight='balanced'
model_weighted = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

results['class_weights'] = train_and_evaluate(
    X_train, y_train, X_test, y_test,
    method_name='Class Weights',
    model=model_weighted
)

print(f"✓ Entrenamiento completado")
print(f"  Umbral óptimo encontrado: {results['class_weights']['best_threshold']:.4f}")

# %% [markdown]
# ## 3. Comparación de Métricas

# %%
# Crear DataFrames comparativos
metrics_comparison_default = pd.DataFrame([
    results[method]['metrics_default'] for method in results.keys()
])

metrics_comparison_optimal = pd.DataFrame([
    results[method]['metrics_optimal'] for method in results.keys()
])

print("\n" + "="*80)
print("COMPARACIÓN DE MÉTRICAS")
print("="*80)

print("\n📊 Métricas con Umbral por Defecto (0.5):")
print("-" * 80)
print(metrics_comparison_default.round(4).to_string(index=False))

print("\n📊 Métricas con Umbral Óptimo (F1-Score maximizado):")
print("-" * 80)
print(metrics_comparison_optimal.round(4).to_string(index=False))

# %% [markdown]
# ## 4. Visualizaciones Comparativas

# %% [markdown]
# ### 4.1 Gráfico de Barras: Comparación de Métricas

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comparación de Métricas entre Estrategias (Umbral Óptimo)', 
             fontsize=16, fontweight='bold')

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    values = [results[method]['metrics_optimal'][metric] for method in results.keys()]
    methods = list(results.keys())
    
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    
    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Eliminar el último subplot vacío
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 Curvas ROC Superpuestas

# %%
plt.figure(figsize=(10, 8))

colors_roc = {'baseline': '#1f77b4', 'oversampling': '#ff7f0e', 
              'undersampling': '#2ca02c', 'class_weights': '#d62728'}

for method_name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    auc = result['metrics_optimal']['auc_roc']
    plt.plot(fpr, tpr, color=colors_roc[method_name], lw=2.5, 
             label=f"{result['metrics_optimal']['method']} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Clasificador Aleatorio')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12, fontweight='bold')
plt.title('Comparación de Curvas ROC', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.3 Matrices de Confusión Comparativas

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Matrices de Confusión (Umbral Óptimo)', fontsize=16, fontweight='bold')

for idx, (method_name, result) in enumerate(results.items()):
    ax = axes[idx // 2, idx % 2]
    cm = confusion_matrix(y_test, result['y_pred_optimal'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Llueve', 'Llueve'],
                yticklabels=['No Llueve', 'Llueve'],
                cbar_kws={'label': 'Cantidad'})
    
    ax.set_ylabel('Valor Real', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicción', fontsize=11, fontweight='bold')
    ax.set_title(f"{result['metrics_optimal']['method']}\n"
                 f"Umbral: {result['best_threshold']:.3f} | "
                 f"F1: {result['metrics_optimal']['f1']:.3f}",
                 fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.4 Comparación Precision vs Recall

# %%
fig, ax = plt.subplots(figsize=(10, 8))

for method_name, result in results.items():
    precision = result['metrics_optimal']['precision']
    recall = result['metrics_optimal']['recall']
    f1 = result['metrics_optimal']['f1']
    
    ax.scatter(recall, precision, s=300, alpha=0.7, 
              color=colors_roc[method_name],
              edgecolors='black', linewidth=2,
              label=f"{result['metrics_optimal']['method']} (F1={f1:.3f})")
    
    # Añadir etiquetas
    ax.annotate(result['metrics_optimal']['method'], 
               (recall, precision),
               textcoords="offset points", xytext=(0,10), 
               ha='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision vs Recall (Umbral Óptimo)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])

# Añadir líneas de referencia de F1
f1_levels = [0.4, 0.6, 0.8]
for f1 in f1_levels:
    x = np.linspace(0.01, 1)
    y = (f1 * x) / (2 * x - f1)
    y = np.where(y > 0, y, np.nan)
    ax.plot(x, y, 'k--', alpha=0.2, linewidth=1)
    ax.text(0.9, (f1 * 0.9) / (2 * 0.9 - f1), f'F1={f1}', 
           fontsize=8, alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Tabla Resumen Final

# %%
print("\n" + "="*80)
print("RESUMEN FINAL Y RECOMENDACIONES")
print("="*80)

# Encontrar el mejor modelo según F1-Score
best_method = max(results.items(), 
                 key=lambda x: x[1]['metrics_optimal']['f1'])

print(f"\n🏆 Mejor modelo según F1-Score: {best_method[1]['metrics_optimal']['method']}")
print(f"   F1-Score: {best_method[1]['metrics_optimal']['f1']:.4f}")
print(f"   Precision: {best_method[1]['metrics_optimal']['precision']:.4f}")
print(f"   Recall: {best_method[1]['metrics_optimal']['recall']:.4f}")
print(f"   AUC-ROC: {best_method[1]['metrics_optimal']['auc_roc']:.4f}")
print(f"   Umbral óptimo: {best_method[1]['best_threshold']:.4f}")

# Tabla resumen
print("\n📋 Tabla Resumen Completa:")
print("-" * 80)
summary = metrics_comparison_optimal.copy()
summary = summary.round(4)
summary = summary.sort_values('f1', ascending=False)
print(summary.to_string(index=False))

print("\n" + "="*80)

# %% [markdown]
# ## 6. Análisis de Trade-offs
#
# ### Interpretación de los Resultados:
#
# - **Sin Corrección (Baseline)**: Puede tener alta accuracy pero pobre recall en la clase minoritaria
# - **Oversampling (SMOTE)**: Genera datos sintéticos, mejora recall pero puede causar overfitting
# - **Undersampling**: Reduce datos de la clase mayoritaria, rápido pero pierde información
# - **Class Weights**: Ajusta la función de costo sin modificar el dataset, buen balance
#
# ### Recomendaciones según el objetivo:
#
# - **Maximizar Recall** (detectar todas las lluvias): Elegir oversampling o undersampling
# - **Maximizar Precision** (evitar falsas alarmas): Elegir el modelo con mayor precision
# - **Balance general**: Elegir el modelo con mayor F1-Score (usualmente class weights o SMOTE)
# - **Interpretabilidad**: Class weights mantiene el dataset original
