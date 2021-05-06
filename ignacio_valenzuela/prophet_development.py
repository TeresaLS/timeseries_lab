# SARIMA Forecast - PM Particles Forecast By Ignacio Valenzuela @ tinamica

# This script will train and test a dataset with a SARIMAX model with a walkforward validation. The parameters can be adjusted in the bottom.

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


def mean_absolute_percentage_error(y_true, y_pred):
    """
This function will calculate the mean absolute percentage error (MAPE) for two desired arrays.
    :param y_true: validation test set
    :param y_pred: predicted data
    :return(float): MAPE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def test_train_spl(data, testsize):
    """
split the dataframe for train and test for the timeseries
    data (pandas Dataframe): dataframe with the values
    testsize (int):number of days to test
    return: two data frames, test and train, respectively.
    """
    test = data.tail(testsize)
    train = data.head(data.shape[0] - testsize)
    return test, train


# this function is defined here to accept any value greater than 0 as 1, just to get the presence of a variable.
def boolean_gen(a):
    if a > 0:
        val = float(1.0)
    else:
        val = float(0.0)
    return val


def mod_prophet(train, test, dependent_var_col, outpath, name, changepoints=None, freq='D', n_changepoints=10, reg_cols=None, country_iso_code='ES', change_scale=0.05):
    """
This function performs the training and testing of a prophet model and returns the main performance metrics
    :param train: dataset with train data
    :param test: dataset with test data. The columns should be the same ones than in the train data
    :param ts_col: name of the column with the objective variable
    :param outpath: path to save the files and plots in
    :param name: name of the series to use when saving the plots
    :param changepoints: list of dates where a break in the series is added manually. Defaults to None.
    :param freq: frequency of the series ('D' for daily, 'W' for weekly, 'M' for monthly). Defaults to daily.
    :param n_changepoints: Number of changepoints to be used in the model. Defaults to 10.
    :param reg_cols: list of names of the columns in the dataframe to be added as regressors in the model.
    :param country_iso_code: country code to use the holidays of each one.
    :param change_scale: rate of learning in the prophet model. Defaults to 0.05
    :return: mae, rmse, mape, name, predictions, conf_intervals
    """
    # path definition
    if name not in os.listdir(outpath):
        os.mkdir(outpath + name)
        print('creating output folder in: \n', outpath + name)
    report_output_path = str(outpath) + str(name) + '/'

    # join both dataframes to plot when the model is done
    train.index = pd.to_datetime(train.index)
    test.index = pd.to_datetime(test.index)
    orig_df = train.append(test)

    if changepoints != None:
        changepoints = list(pd.to_datetime(changepoints))
        changepoints = [date for date in changepoints if date < train.index.max()]
        if len(changepoints) == 0:
            changepoints = None

    if changepoints is None:
        mod = Prophet(n_changepoints=n_changepoints, yearly_seasonality=True, changepoint_prior_scale=change_scale, changepoint_range=0.95, seasonality_mode='additive')
    else:
        mod = Prophet(changepoints=changepoints, yearly_seasonality=True, changepoint_prior_scale=change_scale, changepoint_range=0.95, seasonality_mode='additive')

    mod.add_country_holidays(country_name=country_iso_code)

    if reg_cols is not None:
        for regressor in reg_cols:
            mod.add_regressor(regressor, standardize=False, mode='multiplicative')
            print('adding regressor: ', regressor, '\n')
            reg_cols.append(dependent_var_col)
            cols = reg_cols.copy()
            reg_cols.remove(dependent_var_col)
        tr_df = train[cols].reset_index(drop=False)
    else:
        tr_df = train[[dependent_var_col]].reset_index(drop=False)

    if 'Date' not in list(tr_df.columns):
        tr_df.reset_index(drop=False, inplace=True)
    tr_df = tr_df.rename(columns={'Date':'ds', dependent_var_col: 'y'})

    print(tr_df.head())
    # fit the data
    mod.fit(tr_df)
    # forecast

    future = mod.make_future_dataframe(periods=test.shape[0], freq=freq)
    if reg_cols is not None:
        for column in reg_cols:
            # change after testing
            future[str(column)] = 0

    forecast = mod.predict(future)

    mae = mean_absolute_error(y_pred=forecast['yhat'].tail(test.shape[0]), y_true=test[dependent_var_col])
    rmse = np.sqrt(mean_squared_error(y_pred=forecast['yhat'].tail(test.shape[0]), y_true=test[dependent_var_col]))
    mape = mean_absolute_percentage_error(y_pred=forecast['yhat'].tail(test.shape[0]), y_true=test[dependent_var_col])

    plot = mod.plot(forecast, xlabel='Date', ylabel=dependent_var_col)
    a = add_changepoints_to_plot(plot.gca(), mod, forecast)
    plt.savefig(report_output_path + 'fc_plot_' + name + '.png')
    plt.close('all')

    # components plot
    mod.plot_components(forecast)
    plt.savefig(report_output_path + name + 'components_plot.png')
    plt.close('all')

    deltas = mod.params['delta'].mean(0)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111)
    ax.bar(range(len(deltas)), deltas)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    ax.set_ylabel('Rate change')
    ax.set_xlabel('Potential changepoint')
    fig.tight_layout()

    # Create a text box for the iteration results
    mod.plot_components(forecast)
    fig = plot_plotly(mod, forecast)
    py.plot(fig, filename=report_output_path + name + 'fbprophet_plot.html', auto_open=False)

    aux_res_df = forecast.set_index('ds', drop=True)
    aux_res_df.index = pd.to_datetime(aux_res_df.index)
    aux_res_df = orig_df.merge(aux_res_df, left_index=True, right_index=True)

    predictions = aux_res_df.tail(test.shape[0])
    conf_intervals = forecast[['yhat_lower', 'yhat_upper']]

    aux_res_df = aux_res_df.tail(180)
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.plot(aux_res_df["yhat"], color='darkred', label='Forecast')
    plt.plot(aux_res_df[dependent_var_col], color='darkblue', label='Real')
    plt.plot(aux_res_df['yhat_lower'], color='darkgreen', label='Upper', alpha=0.5)
    plt.plot(aux_res_df['yhat_upper'], color='darkgreen', label='Lower', alpha=0.5)
    ax.fill_between(test.index, aux_res_df['yhat_lower'].tail(test.shape[0]), aux_res_df['yhat_upper'].tail(test.shape[0]), facecolor='darkgreen', alpha=0.2, interpolate=True)
    plt.axvline(x=pd.to_datetime(test.index.min(), format='%Y-%m-%d'), color='grey', linestyle='--')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10000))
    plt.gcf().autofmt_xdate()

    # generate a text box
    props = dict(boxstyle='round', facecolor='white')
    # place a text box in upper left in axes coords

    textstr = 'MAE:' + str(round(mae, 0)) + '\n' + 'MAPE:' + str(round(mape, 2))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.legend()
    plt.savefig(report_output_path + 'pfc_' + name + '_' + str(datetime.strftime(pd.to_datetime(test.index.min()), format='%Y-%m-%d')) + '.png')

    plt.close('all')

    return mae, rmse, mape, name, predictions, conf_intervals


# Lectura de los datos
pm_daily = pd.read_csv(vde_path + 'VDE_daily_madrid_pollution.csv', sep=",")
pm_daily.set_index('Date', inplace=True)
pm_weekly = pd.read_csv(vde_path + 'VDE_weekly_madrid_pollution.csv', sep=",")
pm_weekly.set_index('Date', inplace=True)

mod_prophet(train=pm_daily.head(-15), test=pm_daily.tail(15), dependent_var_col='PM_CENTRO', name ='prophet_01',
            outpath='C:/Users/ignacio.valenzuela/Documents/01. Proyectos Ignacio Valenzuela/Formacion_Conocimiento/TimeseriesLab/timeseries_lab/ignacio_valenzuela/Report/',
            n_changepoints=20,freq='D',change_scale=0.05)
