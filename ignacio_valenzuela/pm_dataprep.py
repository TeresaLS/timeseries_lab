# Timeseries Lab - PM Particles Forecast Data Preparation by Ignacio Valenzuela @ Tinámica

import pandas as pd
import os
import numpy as np


pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.getcwd()

pm_daily=pd.read_csv('timeseries_lab/ignacio_valenzuela/data/VAL_daily_madrid_pollution.csv', sep=';')
pm_daily.set_index('date', inplace=True)
pm_daily.index = pd.to_datetime(pm_daily.index)

pm_weekly=pm_daily.resample('W-MON').agg(
    {'SEASON': 'min', 'PM_RETIRO': 'mean', 'PM_VALLECAS': 'mean', 'PM_CIUDADLINEAL': 'mean', 'PM_CENTRO': 'mean', 'DEW_POINT': 'mean', 'HUMIDITY': 'mean', 'TEMPERATURE': 'mean',
     'WIND_SPEED': 'mean', 'PRECIPITAITON': 'sum', 'COMMULATIVE_PRECIPITATION': 'max'})


def ts_transform(data, ts_col, date_col=None, fill_zeros=False, fill_nan=False, fix_trend=False, lag_differentiation=1, log_transform=False, ma_transform=False, ma_periods=2):
    """
    This function does the basic transformations needed in a typical timeseries according to what is needed in each one
    data (Pandas Dataframe): Dataframe with a date as the index.
    ts_col (str): Time series Column.
    date_col (str): if None, datecol is in the index. Otherwise this should be the string
    fill_nan (bool): True if the NaN should be corrected.
    fix_trend (bool): True if the trend is going to be corrected by differentiation.
    lag_differentiation (int): Number of periods to lag in the differentiation.
    log_transform (bool): Should the ts be transformed to logarithm.
    ma_transform (bool): True if a moving average transform should be done.
    ma_periods (int): Number of periods to apply the moving average, default is 2.


    :returns pandas dataframe with the transformed time series
    """
    if fill_zeros is True:
        data[data[ts_col] == 0] = np.nan
    if fill_nan is True:
        if data[ts_col].isna().sum() != 0:
            print(data[ts_col].isna().sum(), ' values will be interpolated \n', 'Of a total of ', data[ts_col].shape[0])
            data[ts_col] = data[ts_col].interpolate()
        else:
            print('No NaN Values Found')

    if log_transform is True:
        if any(data[ts_col] == 0):
            print('Warning, ', ts_col, ' contains', (data[ts_col] == 0).sum(), ' Zeros, Transformation will be done with log(1+ Value)')
            data[ts_col] = np.log(data[ts_col] + 1)
        else:
            data[ts_col] = np.log(data[ts_col])

    if fix_trend is True:
        data[ts_col] = data[ts_col] - data[ts_col].shift(lag_differentiation)

    if ma_transform is True:
        data[ts_col] = data[ts_col].rolling(ma_periods).mean()
        if fill_nan is True:
            data = data.dropna(subset=[ts_col])

    if date_col is not None:
        data.set_index([date_col], inplace=True)
        print('Index set as ', type(data.index))
    return data[ts_col]


pm_daily['log_pmcentro'] = ts_transform(data=pm_daily, ts_col='PM_CENTRO', fill_nan=True, log_transform=True)
pm_daily['log_dewpoint'] = ts_transform(data=pm_daily, ts_col='DEW_POINT', fill_nan=True, log_transform=True)
pm_daily['log_humidity'] = ts_transform(data=pm_daily, ts_col='HUMIDITY', fill_nan=True, log_transform=True)
pm_daily['log_temperature'] = ts_transform(data=pm_daily, ts_col='TEMPERATURE', fill_nan=True, log_transform=True)
pm_daily['log_wind'] = ts_transform(data=pm_daily, ts_col='WIND_SPEED', fill_nan=True, log_transform=True)
pm_daily['log_precipitation'] = ts_transform(data=pm_daily, ts_col='PRECIPITAITON', fill_nan=True, log_transform=True)
pm_daily['log_cumprecipitation'] = ts_transform(data=pm_daily, ts_col='COMMULATIVE_PRECIPITATION', fill_nan=True, log_transform=True)


pm_weekly['log_pmcentro'] = ts_transform(data=pm_weekly, ts_col='PM_CENTRO', fill_nan=True, log_transform=True)
pm_weekly['log_dewpoint'] = ts_transform(data=pm_weekly, ts_col='DEW_POINT', fill_nan=True, log_transform=True)
pm_weekly['log_humidity'] = ts_transform(data=pm_weekly, ts_col='HUMIDITY', fill_nan=True, log_transform=True)
pm_weekly['log_temperature'] = ts_transform(data=pm_weekly, ts_col='TEMPERATURE', fill_nan=True, log_transform=True)
pm_weekly['log_wind'] = ts_transform(data=pm_weekly, ts_col='WIND_SPEED', fill_nan=True, log_transform=True)
pm_weekly['log_precipitation'] = ts_transform(data=pm_weekly, ts_col='PRECIPITAITON', fill_nan=True, log_transform=True)
pm_weekly['log_cumprecipitation'] = ts_transform(data=pm_weekly, ts_col='COMMULATIVE_PRECIPITATION', fill_nan=True, log_transform=True)

pm_daily = pm_daily.reset_index(drop=False)
pm_daily['Date']=pm_daily['date']
pm_daily['Day of Week'] = pd.to_datetime(pm_daily['date']).apply(lambda time: time.dayofweek)
pm_daily['date'] = pm_daily['date'].map(str)
pm_daily[['year', 'month', 'date']] = pm_daily['date'].str.split('-', expand=True)
pm_daily['month'] = pm_daily['month'].map(int)

pm_weekly = pm_weekly.reset_index(drop=False)
pm_weekly['Date']=pm_weekly['date']
pm_weekly['Day of Week'] = pd.to_datetime(pm_weekly['date']).apply(lambda time: time.dayofweek)
pm_weekly['date'] = pm_weekly['date'].map(str)
pm_weekly[['year', 'month', 'date']] = pm_weekly['date'].str.split('-', expand=True)
pm_weekly['month'] = pm_weekly['month'].map(int)


pm_daily.to_csv('timeseries_lab/ignacio_valenzuela/data/VDE_daily_madrid_pollution.csv')
pm_weekly.to_csv('timeseries_lab/ignacio_valenzuela/data/VDE_weekly_madrid_pollution.csv')