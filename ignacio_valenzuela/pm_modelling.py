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



####################################################
#               TS TRANSFORMATION                  #
####################################################
def ts_transform(data, ts_col, date_col=None, fill_zeros=False, fill_nan=False, fix_trend=False, lag_differentiation=1, log_transform=False, ma_transform=False, ma_periods=2,
                 accumulate_transform=False, accumulation_periods=None):
    """
    This function does the basic transformations needed in a typical timeseries according to what is needed in each one
    data (Pandas Dataframe): Dataframe with a date as the index.
    ts_col (str): Time series Column.
    date_col (str): if None, datecol is in the index. Otherwise this should be the string
    fill_zeros(bool): if zeros will be filled.
    fill_nan (bool): True if the NaN should be corrected.
    fix_trend (bool): True if the trend is going to be corrected by differentiation.
    lag_differentiation (int): Number of periods to lag in the differentiation.
    log_transform (bool): Should the timeseries be transformed to logarithm.
    ma_transform (bool): True if a moving average transform should be done.
    ma_periods (int): Number of periods to apply the moving average, default is 2.


    :returns pandas series with the transformed time series
    """
    df = data.copy(deep=True)

    if fill_zeros is True:
        df[df[ts_col] == 0] = np.nan
    if fill_nan is True:
        if df[ts_col].isna().sum() != 0:
            print(df[ts_col].isna().sum(), ' values will be interpolated \n', 'Of a total of ', df[ts_col].shape[0])
            df[ts_col] = df[ts_col].interpolate()
        else:
            print('No NaN Values Found')

    if log_transform is True:
        if any(df[ts_col] == 0):
            print('Warning, ', ts_col, ' contains ', (df[ts_col] == 0).sum(), ' Zeros, Transformation will be done with log(1+ Value)')
            df[ts_col] = np.log(df[ts_col] + 1)
        else:
            df[ts_col] = np.log(df[ts_col])

    if fix_trend is True:
        df[ts_col] = df[ts_col] - df[ts_col].shift(lag_differentiation)

    if ma_transform is True:
        df[ts_col] = df[ts_col].rolling(ma_periods).mean()
        if fill_nan is True:
            df = df.dropna(subset=[ts_col])

    if accumulate_transform is True:
        if accumulation_periods is None:
            print('Accumulation periods are not defined!')
        df[ts_col] = df[ts_col].rolling(accumulation_periods).sum()

    if date_col is not None:
        df.set_index([date_col], inplace=True)
        print('Index set as ', type(df.index))

    return df[ts_col]


####################################################
#                SARIMAX MODEL                     #
####################################################


def mod_sarima(train, test, dependent_var_col, trend, p, d, q, P, D, Q, S, is_log, outpath, name, xreg, plot_regressors,mle_regression=True, time_varying_regression=False, periodicity='daily'):
    """
This function trains and tests the SARIMA model. for this two dataframes must be given, train and test.
trend, pdq and PDQS, are the statsmodels.SARIMAX variables.
    :param train (Pandas Dataframe): train data
    :param test (Pandas Dataframe): test data
    :param ts_col (int): column of the objective variable
    :param trend (str): Parameter controlling the deterministic trend polynomial A(t)
    :param p (int): Autorregresive parameter
    :param d (int): Differencing parameter
    :param q (int): Differencing Moving Average parameter
    :param P (int): Seasonal Autorregresive parameter
    :param D (int): Seasonal Differencing parameter
    :param Q (int): Seasonal Differencing Moving Average parameter
    :param S (int): Lags for the seasonal
    :param is_log (bool): true if the series is in logarithm. defaults to False.
    :param outpath (str): path where the results will be stored
    :param name (str): name to use when saving the files returned by the model
    :xreg(list): list of strings with names of columns in the test/train datasets to be used as regressors
    :plot_regressors: whether the regressors should be plotted in the function
    :return: mae_error (float): Mean Absolute Error
    rmse_error (float): root mean squared error
     res_df (Pandas Dataframe): Dataframe with all data and the prediction in the Forecast column.
      mod (statsmodel object): Model object.
    """
    print('Modelling \n', name, ' Forecast - SARIMAX ' + '(' + str(p) + ',' + str(d) + ',' + str(q) + ')' + 'S' + '(' + str(P) + ',' + str(D) + ',' + str(Q) + ')' + str(S))

    # path definition
    if name not in os.listdir(outpath):
        os.mkdir(outpath + name)
        print('creating output folder in: \n', outpath + name)
    report_output_path = str(outpath) + str(name) + '/'

    # fit the model
    if len(xreg) == 0:
        mod = SARIMAX(train[dependent_var_col], trend=trend, order=(p, d, q), seasonal_order=(P, D, Q, S), time_varying_regression=time_varying_regression, mle_regression =mle_regression ).fit()
    else:
        mod = SARIMAX(train[dependent_var_col], trend=trend, order=(p, d, q), seasonal_order=(P, D, Q, S), exog=train[xreg], enforce_stationarity=False,
                      time_varying_regression=time_varying_regression, mle_regression =mle_regression ).fit()

    # plot diagnostics
    plt.figure()
    plt.title('Plot diagnostics for' + dependent_var_col + ' Forecast - SARIMA '
              + '(' + str(p) + ',' + str(d) + ',' + str(q) + ')' + 'S' + '(' + str(P) + ',' + str(D) + ',' + str(Q) + ')' + str(S))
    mod.plot_diagnostics(figsize=(15, 9), lags=40)
    plt.savefig(report_output_path + 'diagnostics_' + name + '.png')

    # predict with the model
    # I know this seems like a lot, but to be able to support broken time series in the forecast you need to reset the indexes

    test_aux = test.copy(deep=True)

    # TODO: remove this parameter
    test_aux[xreg]=np.exp(test_aux[xreg])
    test_aux[xreg]=test_aux[xreg]*0.9
    test_aux[xreg]=np.log(test_aux[xreg])


    test_aux.reset_index(drop=True, inplace=True)
    train_aux = train.copy(deep=True)
    train_aux.reset_index(drop=True, inplace=True)

    # get the predictions with the model
    if len(xreg) == 0:
        predictions = mod.predict(train_aux.index.max() + 1, end=train_aux.index.max() + 1 + test_aux.index.max())
        conf_intervals = mod.get_prediction(train_aux.index.max() + 1, end=train_aux.index.max() + 1 + test_aux.index.max()).conf_int(alpha=0.5)
    else:
        predictions = mod.predict(train_aux.index.max() + 1, end=train_aux.index.max() + 1 + test_aux.index.max(), exog=test_aux[xreg])
        conf_intervals = mod.get_prediction(train_aux.index.max() + 1, end=train_aux.index.max() + 1 + test_aux.index.max(), exog=test_aux[xreg]).conf_int(alpha=0.5)

    predictions.index = test.index
    conf_intervals.index = test.index

    # the confidence interval is trimmed for extreme values so they don't overextort after missing dates and doing the inverse log transf (exp)
    conf_intervals = pd.DataFrame(conf_intervals)
    # conf_intervals[(conf_intervals['lower log_revenue_emi'] < conf_intervals['lower log_revenue_emi'].quantile(q=0.01)) | (
    #         conf_intervals['upper log_revenue_emi'] > conf_intervals['upper log_revenue_emi'].quantile(q=0.99))] = np.nan

    conf_intervals.index = conf_intervals.index.date
    conf_intervals.index = conf_intervals.index.map(str)

    # assign the predictions to the test dataframe to be used later in the plotting
    test['Forecast'] = predictions
    train['Forecast'] = mod.fittedvalues

    # add the columns that are in the regressors to the dataframe that will be used and get a dataframe to plot (train aux)
    columns = [dependent_var_col, 'Forecast']
    columns.append(xreg)
    columns = list(flatten(columns))
    train_aux = train[columns]
    test_aux = test[columns]
    test_aux = pd.merge(test_aux, conf_intervals, left_index=True, right_index=True)

    # transform the data back from logarithm if the series is in that scale
    if is_log is True:
        res_df = pd.concat([train_aux, test_aux])
        res_df['Forecast'] = np.exp(res_df['Forecast'])
        res_df[dependent_var_col] = np.exp(res_df[dependent_var_col])

        mae_error = mean_absolute_error(np.exp(test[dependent_var_col]), np.exp(predictions))
        rmse_error = np.sqrt(mean_squared_error(np.exp(test[dependent_var_col]), np.exp(predictions)))
        mape = mean_absolute_percentage_error(np.exp(test[dependent_var_col]), np.exp(predictions))

        preds=np.exp(predictions)

    else:
        res_df = pd.concat([train_aux, test_aux])
        mae_error = mean_absolute_error(test[dependent_var_col], predictions)
        rmse_error = np.sqrt(mean_squared_error(test[dependent_var_col], predictions))
        mape = mean_absolute_percentage_error(test[dependent_var_col], predictions)
        preds=predictions

    # Create a text box for the iteration results
    textstr = 'MAE:' + str(round(mae_error, 0)) + '\n' + 'MAPE:' + str(round(mape, 2))

    aux_res_df = res_df.tail(365)  # only plot the 6 months
    aux_res_df.index = pd.to_datetime(aux_res_df.index)
    if str(periodicity).upper() is 'daily':
        aux_res_df = aux_res_df.reindex(pd.date_range(aux_res_df.index.min(), aux_res_df.index.max()), fill_value=np.nan)

    # Upper and lower confidence intervals
    lower = aux_res_df[str('lower ' + str(dependent_var_col))]
    upper = aux_res_df[str('upper ' + str(dependent_var_col))]
    if is_log is True:
        lower = np.exp(lower)
        upper = np.exp(upper)

    # plot the figure with the prediction
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.subplots_adjust(right=0.85, left=0.05, bottom=0.1)
    ax2 = ax.twinx()
    ax.plot(aux_res_df["Forecast"], color='darkred', label='Forecast')
    ax.plot(aux_res_df[dependent_var_col], color='darkblue', label='Real')
    if plot_regressors is True:
        for i in xreg:
            ax2.plot(aux_res_df[i], color='grey', alpha=0.4, label=str(i))
    ax.plot(lower, color='darkgreen', label='Lower', alpha=0.5)
    ax.plot(upper, color='darkgreen', label='Upper', alpha=0.5)
    ax.fill_between(upper.dropna().index, upper.dropna(), lower.dropna(), facecolor='darkgreen', alpha=0.2, interpolate=False)
    ax.axvline(x=pd.to_datetime(test.index.min(), format='%Y-%m-%d'), color='grey', linestyle='--')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    plt.gcf().autofmt_xdate()
    # generate a text box
    props = dict(boxstyle='round', facecolor='white')
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    ax.legend(title='Forecast Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.legend(title='Regressors', bbox_to_anchor=(1.05, 0.7), loc='center left')
    plt.savefig(report_output_path + 'Forecast_' + name + '_' + str(datetime.strftime(pd.to_datetime(test.index.min()), format='%Y-%m-%d')) + '.png')
    plt.title('SARIMAX Forecast of ' + name)
    plt.show()

    plt.close('all')

    # plotting the results in plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df[dependent_var_col],
                             mode='lines',
                             name='Real'))
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['Forecast'],
                             mode='lines+markers',
                             name='Fitted - Forecasted'))

    fig.add_shape(
        dict(type="line",
             x0=test.index.min(),
             y0=res_df[dependent_var_col].min(),
             x1=test.index.min(),
             y1=res_df[dependent_var_col].max(),
             line=dict(
                 color="grey",
                 width=1)))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title=dependent_var_col + ' Forecast - SARIMA ' + '(' + str(p) + ',' + str(d) + ',' + str(q) + ')' + 'S' + '(' + str(P) + ',' + str(D) + ',' + str(Q) + ')' + str(S),
        xaxis_title=dependent_var_col,
        yaxis_title='Date',
        font=dict(
            family="Century gothic",
            size=18,
            color="darkgrey"))
    fig.write_html(report_output_path + name + '_forecast_SARIMA.html')
    plt.close('all')

    print('MAE', mae_error)
    print('RMSE', rmse_error)
    print('MAPE', mape)
    print(mod.summary())



    return mae_error, rmse_error, mape, name, preds, conf_intervals


####################################################
#                PROPHET MODEL                     #
####################################################


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

    if 'DATE' not in list(tr_df.columns):
        tr_df.reset_index(drop=False, inplace=True)
    tr_df = tr_df.rename(columns={'DATE': 'ds', 'index': 'ds', dependent_var_col: 'y'})

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


####################################################
#         WALKFORWARD VALIDATION                   #
####################################################


def walkforward_validation(data, test_start_date, test_end_date=None, step_size=15, testsize=15, model='SARIMA'):
    """
This function performs a walkforward validation of the model. This means that the model will be trained with all available data until the breakpoint and tested testsize days.

    :param data: dataframe with all the test and train data
    :param test_start_date: date when test is starting
    :param test_end_date: date when there won't be any more tests, defaults to last value of the dataset
    :param step_size: by how many dates a test should be done.
    :param testsize: size of the test to use.
    :param model: SARIMA or PROPHET model to be used
    :return: modelling_results
    """
    test_start_date = pd.to_datetime(test_start_date)
    current_max_date = test_start_date

    modelling_results = pd.DataFrame(columns=['series_name', 'model_type', 'test_start', 'test_end', 'MAE', 'MAPE', 'RMSE'])

    if test_end_date is None:
        test_end_date = data.index.max()
        test_end_date = pd.to_datetime(test_end_date)
    else:
        test_end_date = pd.to_datetime(test_end_date)

    while current_max_date < test_end_date:
        data.index = pd.to_datetime(data.index)
        iter_data = data[data.index <= current_max_date + timedelta(days=testsize)]
        test, train = test_train_spl(iter_data, testsize=testsize)

        if (model.upper() == 'SARIMA') | (model.upper() == 'SARIMAX'):
            print('USING SARIMA MODEL')
            mae, rmse, mape, name, preds, conf_intervals = mod_sarima(train=train, test=test, **arima_model_params)
        elif model.upper() == 'PROPHET':
            print('USING PROPHET MODEL')
            mae, rmse, mape, name, preds, conf_intervals = mod_prophet(train=train, test=test, **prophet_model_params)
        else:
            print('model name not known')
        iter_results = pd.DataFrame({'series_name': name, 'model_type': model, 'test_start': [current_max_date],
                                     'test_end': [current_max_date + timedelta(testsize)], 'MAE': [mae], 'MAPE': [mape], 'RMSE': [rmse]})
        modelling_results = modelling_results.append(iter_results, ignore_index=True)

        # this line is just for validation of the effect of regressors in the forecast
        preds.to_csv(mod_report_path + arima_model_params['name'] + 'forecast_' + str(current_max_date).replace(':', '')+ '.csv')

        current_max_date = current_max_date + timedelta(days=step_size)

    return modelling_results



# Lectura de los datos
pm_daily=pd.read_csv(vde_path+'VDE_daily_madrid_pollution.csv', sep=",")
pm_daily.set_index('Date', inplace=True)
pm_weekly=pd.read_csv(vde_path+'VDE_weekly_madrid_pollution.csv', sep=",")
pm_weekly.set_index('Date', inplace=True)


arima_model_params = dict(dependent_var_col='log_pmcentro', trend='n', p=1, d=1, q=2, P=0, D=0, Q=0, S=0, is_log=True, outpath=mod_report_path,
                    name='pm_daily_log1120000_hum_pre', time_varying_regression=False,mle_regression=True,
                    xreg=['log_humidity','COMMULATIVE_PRECIPITATION'],
                    plot_regressors=True, periodicity=365)

# Prophet Model
prophet_model_params = dict(changepoints=None, n_changepoints=20, change_scale=0.5, dependent_var_col='PM_CENTRO', outpath=mod_report_path, name='pm_daily', freq='D',
                            reg_cols=[], country_iso_code='ES')

walkforward_validation_params = dict(data=pm_daily, test_start_date='2015-11-01', test_end_date=None, step_size=15, testsize=5, model='sarimax')

walkforward_validation(**walkforward_validation_params)

for p in [0,1,2,3]:
    for d in [0,1,2]:
        for q in [0,1,2]:
            arima_model_params = dict(dependent_var_col='log_pmcentro', trend='n', p=p, d=d, q=q, P=0, D=0, Q=0, S=0, is_log=True, outpath=mod_report_path,
                                      name='pm_daily_log5140000', time_varying_regression=False, mle_regression=True,
                                      xreg=['log_humidity'],
                                      plot_regressors=True, periodicity=365)
            wfr=walkforward_validation(**walkforward_validation_params)
            wfr.to_csv(mod_report_path+str(p)+str(d)+str(q)+'_'+str(wfr.MAPE.mean())+'.csv')

