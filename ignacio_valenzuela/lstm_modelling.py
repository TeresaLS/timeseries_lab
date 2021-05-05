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


# supress copy warnings
pd.options.mode.chained_assignment = None
# display parameters
pd.set_option('display.max_rows', 4000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# data paths
vde_path = 'timeseries_lab/ignacio_valenzuela/data/'
mod_report_path = 'timeseries_lab/ignacio_valenzuela/Report/'



# Lectura de los datos
pm_daily=pd.read_csv(vde_path+'VDE_daily_madrid_pollution.csv', sep=",")
pm_daily.set_index('Date', inplace=True)
pm_weekly=pd.read_csv(vde_path+'VDE_weekly_madrid_pollution.csv', sep=",")
pm_weekly.set_index('Date', inplace=True)
