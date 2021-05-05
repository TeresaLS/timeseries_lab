# Timeseries Lab - PM Particles Forecast Analysis by Ignacio Valenzuela @ Tinámica

# Get the relevant libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

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
def random_forest_features_importance(data, dependent_variable, selected_features, n_estimators=100, max_depth=5, max_features=10, show=False):
    selected_features = data[selected_features]
    mod = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
    mod.fit(y=data[dependent_variable], X=selected_features)

    print('Importancia de las variables : \n', mod.feature_importances_)

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
#   Visualization, Autocorrelation and Stationariety
################################################


def ts_analyzer(data, ts_col, gen_outpath, date_col='DATE', name=None, report_export=True):
    """
This function analizes a timeseries in a dataframe, including visualization, decomposition, acf, pacf, dickey-fuller with autolags and moving averages and moving sdev.

    :param data (Pandas Dataframe): a dataframe with unique dates as the index
    :param ts_col (str): str of the name of the column to be named
    :param gen_outpath (str): file path to a general folder where results will be written
    :param date_col (str): date column name
    :param name (str): name of the series, i.e. each country to be analysed
    :param report_export (bool): should the report be written in a separate folder
    """
    # warnings
    if any(data[ts_col].isna()):
        print('Analysis does not work with NA values, correction is needed before continuing the analysis')

    # folder location and outpath redefinition
    os.getcwd()
    os.chdir(gen_outpath)
    if report_export is True:
        if 'REPORT_' + str(ts_col) not in os.listdir():
            os.mkdir('REPORT_' + ts_col + '/')
        if 'SUBREPORT_' + name not in os.listdir('REPORT_' + ts_col + '/'):
            os.mkdir('REPORT_' + str(ts_col) + '/' + 'SUBREPORT_' + str(name) + '/')
    outpath = 'REPORT_' + ts_col + '/' + 'SUBREPORT_' + str(name) + '/'

    # series visualization
    fig = px.line(data, y=ts_col)
    fig.update_xaxes(rangeslider_visible=True)
    if report_export is True:
        fig.write_html(outpath + 'TS.html')

    # Visualization as png instead of html
    fig, axes = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(15)
    plt.suptitle(name + ' - ' + ts_col + ' Timeseries')

    # this is necessary to add because of a bug in matplotlib.
    # plt.rcParams['date.epoch'] = '0000-12-31T00:00:00'

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=90))

    axes.plot(data[ts_col], 'darkblue', label='Residual - Noise')
    axes.legend(loc='upper left')
    axes.xaxis.set_major_locator(ticker.MultipleLocator(90))
    plt.gcf().autofmt_xdate()
    if name is not None:
        fig.savefig(outpath + str(name) + '_timeseries.png')
    else:
        fig.savefig(outpath + '_timeseries.png')
    plt.close('all')

    # series acf and pacf plots
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    plot_acf(data[ts_col], lags=50, ax=axes[0])
    plot_pacf(data[ts_col], lags=50, ax=axes[1])
    if name is not None:
        fig.savefig(outpath + str(name) + 'ACF_PACF.png')
    else:
        fig.savefig(outpath + 'ACF_PACF.png')
    plt.close('all')
    # series dickey fuller test

    adf = adfuller(data[ts_col], autolag='AIC')
    pd.DataFrame(adf).to_csv(outpath + 'adfuller_' + name + '.csv')
    print('The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root')
    print('ADF Statistic: %f' % adf[0])
    print('p-value: %f' % adf[1])

    # Mean, variance plots
    # plt.rcParams['date.epoch'] = '0000-12-31T00:00:00'
    fig, ax = plt.subplots(figsize=(15, 8))
    mv_av = data[ts_col].rolling(window=4).mean()
    plt.suptitle('Moving Averages for Mean and Standard Deviation')
    ax.plot(mv_av, color='darkred', label='Monthly Moving Average')
    mv_std = data[ts_col].rolling(window=4).std()
    ax.plot(mv_std, color='darkblue', label='Monthly Moving Standard Deviation')
    if name is not None:
        fig.savefig(outpath + str(name) + '_monthly_m_avg.png')
    else:
        fig.savefig(outpath + 'monthly_m_avg.png')
    fig.show()
    plt.close('all')

    if name is not None:
        print(str(name), ' analysis is done.')


################################################
#               SEASONALITY
################################################

def ts_seasonality(data, ts_col, periodicity, hue, savefig=False, outpath=None, name=None):
    # Yearly Seasonality
    plt.figure()
    sns.lineplot(data=data, x=periodicity, y=ts_col, hue=hue, legend='full')
    plt.title(name)
    if savefig is True:
        plt.savefig(outpath + name + 'seasonalplot.png')

    # series decomposition
    if any(data[ts_col] == 0):
        print('Multiplicative decomposition is not available with value 0 in the timeseries')
        modeltypes = ['additive']
    else:
        modeltypes = ['additive', 'multiplicative']

    for modeltype in modeltypes:
        if len(data[ts_col]) < 365 * 2:
            print('The series is not long enough to do a decomposition, minimum is 2 seasons')
        else:
            deco = seasonal_decompose(data[ts_col], model=modeltype, period=365)
            estimated_seasonal = deco.seasonal
            estimated_trend = deco.trend
            estimated_residual = deco.resid

            fig, axes = plt.subplots(3, 1, sharex=True, sharey=False)
            fig.set_figheight(8)
            fig.set_figwidth(15)
            plt.suptitle('Decomposition of ' + ts_col + ' with ' + modeltype + ' model')

            # plt.rcParams['date.epoch'] = '0000-12-31T00:00:00'

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=90))

            axes[0].plot(estimated_seasonal, 'darkblue', label='Seasonality')
            axes[0].legend(loc='upper left')

            axes[1].plot(estimated_trend, 'green', label='Trend')
            axes[1].legend(loc='upper left')

            axes[2].plot(estimated_residual, 'darkred', label='Residual - Noise')
            axes[2].legend(loc='upper left')
            axes[2].xaxis.set_major_locator(ticker.MultipleLocator(90))

            plt.gcf().autofmt_xdate()
            if name is not None:
                fig.savefig(outpath + str(name) + 'A' + modeltype + '.png')
            else:
                fig.savefig(outpath + 'dec' + modeltype + '.png')
        plt.close('all')

        # series acf and pacf plots
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        plot_acf(data[ts_col], lags=110, ax=axes[0])
        plot_pacf(data[ts_col], lags=110, ax=axes[1])
        if name is not None:
            fig.savefig(outpath + str(name) + 'ACF_PACF.png')
        else:
            fig.savefig(outpath + 'ACF_PACF.png')
        plt.close('all')


# data paths
vde_path = 'timeseries_lab/ignacio_valenzuela/data/'
mod_report_path = 'C:/Users/ignacio.valenzuela/Documents/01. Proyectos Ignacio Valenzuela/Formacion_Conocimiento/TimeseriesLab/timeseries_lab/ignacio_valenzuela/Report/'

pm_daily = pd.read_csv(vde_path + 'VDE_daily_madrid_pollution.csv')
pm_daily['Date']=pd.to_datetime(pm_daily['Date'])

os.getcwd()
ts_analyzer(data=pm_daily, ts_col='PM_CENTRO', gen_outpath=mod_report_path, date_col='Date', name='pm_centro', report_export=True)
ts_seasonality(data=pm_daily, ts_col='PM_CENTRO', periodicity='month', hue='year', savefig=True, outpath=mod_report_path, name='pm_daily_s_')
random_forest_features_importance(data=pm_daily, dependent_variable='PM_CENTRO',
                                  selected_features=['HUMIDITY', 'TEMPERATURE', 'WIND_SPEED', 'PRECIPITAITON', 'COMMULATIVE_PRECIPITATION'], n_estimators=100, max_depth=5,
                                  max_features=10, show=False)

ts_crosscorrelation(data=pm_daily[pm_daily.index > pd.to_datetime('2013-01-01')], first_col='PM_CENTRO', second_col='WIND_SPEED', diff=True, m_lags=20, plot=True)
