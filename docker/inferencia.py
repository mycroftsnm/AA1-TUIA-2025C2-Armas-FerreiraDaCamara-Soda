import joblib
import pandas as pd
import warnings
import sklearn
import tensorflow as tf

from custom_imputer import CustomImputer
from custom_transformers import (ClimateTransformer, RainySeasonTransformer,
                          LogTransformer, TempDiffTransformer, 
                          CloudScalerTransformer, WindCyclicalTransformer)

print(sklearn.__version__)
warnings.simplefilter('ignore')

import logging
from sys import stdout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

# Lista con el orden necesario de las features para keras
features_order = joblib.load('features_order.joblib')

pipeline = joblib.load('pipeline.joblib')

logger.info('loaded pipeline')

df_input = pd.read_csv('./data/input.csv')

logger.info('loaded input')

input_transformed = pipeline.transform(df_input)

# Obtener el nombre de las columnas
ultimo_paso_real = pipeline.named_steps['fe-scaler'].named_steps['scale']
features_names = ultimo_paso_real.get_feature_names_out()

df_input = pd.DataFrame(input_transformed, columns=features_names)
df_input = df_input[features_order]

model = tf.keras.models.load_model('nn_optimizada.keras')

y_pred_proba = model.predict(df_input)
y_pred = (y_pred_proba > 0.44).astype(int)

logger.info(f'made predictions')

df_output = pd.DataFrame({
    'Proba': y_pred_proba.flatten(),
    'RainTomorrow': y_pred.flatten()
})

df_output.to_csv('./data/predictions.csv', index=False)
logger.info('Saved output to predictions.csv')

