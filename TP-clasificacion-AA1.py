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
#     display_name: Python (myenv)
#     language: python
#     name: myenv
# ---

# %%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# %%
# Carga el dataset en un dataframe
df = pd.read_csv('weatherAUS.csv')

# Revisa si hay filas duplicadas
df.duplicated().sum() # 0 filas duplicadas

# Revisa las columnas y sus tipos de datos
df.dtypes

# %% [markdown]
# # Limpieza y preprocesamiento

# %%
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0}).astype('Int8')
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0}).astype('Int8')

# %%
df['Cloud3pm'].value_counts(dropna=False)

# %%
df['Cloud9am'].value_counts(dropna=False)

# %% [markdown]
# Por el rango de valores que asumen las variables **Cloud9am** y **Cloud3pm** asumimos que dichas variables están medidas en octas, que es la unidad de medida empleada para describir la nubosidad observable en un determinado lugar. https://es.wikipedia.org/wiki/Octa
