import pandas as pd
import numpy as np
import pandas_profiling
import datetime

from dateutil.parser import parse


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns

# Leyendo el conjunto de datos fuentes
madrid_polution_semana_ts=pd.read_csv('C:/TINAMICA/TINAMICA_SPAIN/Laboratorios/Lab_time_series/GitHub_Repositorio/jesus_pernalete/datos/madrid_polution_semana_ts.csv', sep=',')


plt.figure()
plt.title('Serie PM Centro Semana/Año')
sns.lineplot(data=madrid_polution_semana_ts, x='Week', y='PM_CENTRO', hue='YEAR', legend='full')

plt.figure()
plt.title('Box Plot PM Centro Semana/Año')
sns.catplot(data=madrid_polution_semana_ts, x='YEAR', y='PM_CENTRO', kind='box')
