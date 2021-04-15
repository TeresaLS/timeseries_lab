# Timeseries Lab - PM Particles Forecast Analysis by Ignacio Valenzuela @ Tinámica

# Get the relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from datetime import timedelta, datetime
from pandas.core.common import flatten
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as mticker
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot
import plotly.offline as py
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.getcwd()

####################################################
#            Cross Correlation Test                #
####################################################
def ts_crosscorrelation(data, first_col, second_col, diff=True, m_lags=None, plot=True):
    aux_df = data.copy()

    if diff is True:
        aux_df[first_col] = aux_df[first_col].pct_change()
        aux_df[second_col] = aux_df[second_col].pct_change()
        aux_df = aux_df.replace([np.inf, -np.inf], np.nan)
        aux_df = aux_df.dropna(subset=[first_col, second_col])

    if diff is True:
        print('Lag 0 pearson correlation differenciated (pct change): ', pearsonr(aux_df[first_col], aux_df[second_col]))
    else:
        print('Lag 0 pearson correlation NOT differenciated: ', pearsonr(aux_df[first_col], aux_df[second_col]))

    if m_lags is None:
        m_lags = int(len(aux_df[first_col]) / 10)
        if m_lags < 20:
            m_lags = 20

    if plot is True:
        fig = plt.figure(figsize=(15, 10))
        plt.xcorr(aux_df[first_col], aux_df[second_col], maxlags=m_lags)
        plt.title('Cross Correlation Plot Between ' + first_col + ' and ' + second_col)
        fig.show()

    lags, corr, lines, b = plt.xcorr(aux_df[first_col], aux_df[second_col], maxlags=m_lags)
    res = pd.DataFrame(lags, columns=['lags'])
    res['corr'] = corr

    return res


##############################################
#   RANDOM FOREST
##############################################
def random_forest_features_importance(data,dependent_variable , selected_features, n_estimators=100, max_depth=5, max_features=10, show=False):
    selected_features=data[selected_features]
    mod = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    mod.fit(data[dependent_variable], data[selected_features])

    print('Importancia de las variables : \n',mod.feature_importances_)

    importance = pd.DataFrame(mod.feature_importances_, index=data[selected_features].columns, columns=['relative_importance'])

    if show is True:
        # plot feature importance
        plt.figure(figsize=(15, 15))
        plt.bar(importance.index, height=importance['relative_importance'])
        plt.title('Random Forest Model Relative importance for variables')
        plt.xticks(rotation=30)
        plt.show()

    return importance

################################################
#               AUTOCORRELATION
################################################









################################################
#               SEASONALITY
################################################











################################################
#               STATIONARIETY
################################################