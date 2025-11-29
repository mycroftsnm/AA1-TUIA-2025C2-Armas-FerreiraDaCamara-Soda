import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.wind_cols = ['WindDir9am', 'WindDir3pm', 'WindGustDir']

        self.features_imputables = [
            'MinTemp','MaxTemp','Rainfall',
            'Evaporation','Sunshine','WindGustDir',
            'WindGustSpeed','WindDir9am','WindDir3pm',
            'WindSpeed9am','WindSpeed3pm','Humidity9am',
            'Humidity3pm','Pressure9am','Pressure3pm',
            'Cloud9am','Cloud3pm','Temp9am','Temp3pm',
        ]
        
        self.maps_location_median_ = {} # Numéricas: Mediana Location
        self.maps_climate_mean_ = {}    # Numéricas: Media Climate
        self.maps_climate_mode_ = {}    # Categóricas: Moda Climate
        
    def fit(self, X, y=None):
        df = X.copy()
        
        for col in self.features_imputables:
            if col in self.wind_cols:
                self.maps_climate_mode_[col] = df.groupby('Climate')[col].apply(
                    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
                )
            
            else:
                self.maps_location_median_[col] = df.groupby('Location')[col].median()                
                self.maps_climate_mean_[col] = df.groupby('Climate')[col].mean()                
                
        return self

    def transform(self, X):
        X = X.copy()
        
        for col in self.features_imputables:
            if X[col].isna().sum() == 0:
                continue

            if col in self.wind_cols:
                # 1. Intenta imputar con valor de otra hora del mismo registro
                others = [c for c in self.wind_cols if c != col]
                for other_col in others:
                    X[col] = X[col].fillna(X[other_col])
                
                # 2. Intenta imputar con moda histórica del tipo de clima
                fallback_climate = X['Climate'].map(self.maps_climate_mode_[col])
                X[col] = X[col].fillna(fallback_climate)
                
            else:
                # 1. Intenta imputar por mediana histórica de la ubicación
                fallback_location = X['Location'].map(self.maps_location_median_[col])
                X[col] = X[col].fillna(fallback_location)
                
                # 2. Intenta imputar por media histórica del tipo de clima
                fallback_climate = X['Climate'].map(self.maps_climate_mean_[col])
                X[col] = X[col].fillna(fallback_climate)
                
                # Redondeo para mantener en octas
                if col in ['Cloud9am', 'Cloud3pm']:
                    X[col] = X[col].round()

        return X