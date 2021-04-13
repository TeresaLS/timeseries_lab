import pandas as pd
from datetime import timedelta
import numpy as np


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


def mod_prophet():
    # insert the model here :)
    mod_results = []
    return mod_results


def mod_sarima():
    # insert the model here :)
    mod_results = []
    return mod_results


def walkforward_validation(data, test_start_date='2020-07-01', test_end_date=None, step_size=15, testsize=15, model='SARIMA'):
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
            mae, rmse, mape, name, preds, conf_intervals = mod_sarima(train=train, test=test, **sarimax_params)
        elif model.upper() == 'PROPHET':
            print('USING PROPHET MODEL')
            mae, rmse, mape, name, preds, conf_intervals = mod_prophet(train=train, test=test, **prophet_model_params)
        else:
            print('model name not known')
        # se obtiene el resultado de la iteración
        iter_results = pd.DataFrame({'series_name': name, 'model_type': model, 'test_start': [current_max_date],
                                     'test_end': [current_max_date + timedelta(testsize)], 'MAE': [mae], 'MAPE': [mape], 'RMSE': [rmse]})
        modelling_results = modelling_results.append(iter_results, ignore_index=True)

        current_max_date = current_max_date + timedelta(days=step_size)

    return modelling_results


mod_res = walkforward_validation()

# Para evitar tener que poner tantos parámetros usamos dicts, luego se pueden sacar del código
walkforward_validation_params = dict(test_start_date='2019-09-01', test_end_date=None, step_size=15, testsize=15, model='sarimax')

sarimax_params = dict()

prophet_model_params = dict()
