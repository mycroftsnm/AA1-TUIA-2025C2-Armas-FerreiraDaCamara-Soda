import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DIR_ANGULOS = {
    'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5, 'E': 90.0,
    'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5, 'S': 180.0, 'SSW': 202.5,
    'SW': 225.0, 'WSW': 247.5, 'W': 270.0, 'WNW': 292.5, 'NW': 315.0,
    'NNW': 337.5
}

LOCATION_KOPPEN = {
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

class ClimateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Climate'] = X['Location'].map(LOCATION_KOPPEN)
        X['ClimateArid'] = np.where(X['Climate'] == 'Arid', 1, 0)
        X['ClimateTropical'] = np.where(X['Climate'] == 'Tropical', 1, 0)

        return X

class RainySeasonTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if not np.issubdtype(X['Date'].dtype, np.datetime64):
            X['Date'] = pd.to_datetime(X['Date'])
        
        month = X['Date'].dt.month
        tropical_months = {12, 1, 2, 3}
        temperate_months = {6, 7, 8, 9}
        
        cond_tropical = (X['Climate'] == 'Tropical') & (month.isin(tropical_months))
        cond_temperate = (X['Climate'] == 'Temperate') & (month.isin(temperate_months))
        
        X['RainySeason'] = np.where(cond_tropical | cond_temperate, 1, 0)
        return X.drop(columns=['Date'])

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[f'{col}_log'] = np.log1p(X[col])
        return X

class TempDiffTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TempDiff'] = X['MaxTemp'] - X['MinTemp']
        return X

class CloudScalerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Cloud9am'] = X['Cloud9am'] / 8
        X['Cloud3pm'] = X['Cloud3pm'] / 8
        return X

class WindCyclicalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            angulos = X[col].map(DIR_ANGULOS)
            X[f'{col}_sin'] = np.sin(angulos * 2 * np.pi / 360)
            X[f'{col}_cos'] = np.cos(angulos * 2 * np.pi / 360)
            X = X.drop(columns=[col])
        return X