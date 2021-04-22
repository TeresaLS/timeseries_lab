# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:32:34 2021

@author: Jesus Pernalete
""" 

# en el promt se instalo pip install pandas-profiling
# Laboratorio series de tiempo tinamica 

import pandas as pd
import numpy as np
import pandas_profiling
import datetime

# Leyendo el conjunto de datos fuentes
madrid_polution=pd.read_csv('C:/TINAMICA/TINAMICA_SPAIN/Laboratorios/Lab_time_series/GitHub_Repositorio/data/Madrid_polution_dataframe.csv', sep=',')

#En la revisión externa del archivo .csv se evidencia que de las variables PM, la PM_CENTRO,tiene mas datos, en cuanto a la dimensión de tiempo, viene detalle hasta el nivel hora, para simplificar
# la predicción se agrupara a nivel de dia usando el promedio de la variable.

madrid_polution.info()

#Creando la variable fecha, uniendo YEAR+MONTH+DAY
madrid_polution['FECHA']= pd.to_datetime(madrid_polution[['YEAR', 'MONTH', 'DAY']])
madrid_polution['DATE']=pd.to_datetime(madrid_polution[['YEAR', 'MONTH', 'DAY']])

madrid_polution.dtypes

madrid_polution['YM'] = madrid_polution['DATE'].dt.strftime('%Y-%m')



madrid_polution.info()
#Creando un index con la variable fecha
madrid_polution = madrid_polution.set_index('FECHA')
madrid_polution.info()


madrid_polution_diario = madrid_polution.groupby('FECHA', as_index=True).agg(    
                                            SEASON=('SEASON','min'), 
                                            
                                            PM_RETIRO= ('PM_RETIRO','mean'),
                                            PM_VALLECAS= ('PM_VALLECAS','mean'),
                                            PM_CIUDADLINEAL= ('PM_CIUDADLINEAL','mean'),
                                            PM_CENTRO= ('PM_CENTRO','mean'),
                                            
                                            DEW_POINT= ('DEW_POINT','mean'),
                                            HUMIDITY= ('HUMIDITY','mean'),
                                            PREASSURE= ('PREASSURE','mean'),
                                            TEMPERATURE= ('TEMPERATURE','mean'),
                                            WIND_SPEED= ('WIND_SPEED','mean'),
                                            COMMULATIVE_PRECIPITATION= ('COMMULATIVE_PRECIPITATION','max'))

# Automatizar el análisis exploratorio de datos con Pandas-Profiling
from pandas_profiling import ProfileReport

prof = ProfileReport(madrid_polution_diario)
prof.to_file(output_file='C:/TINAMICA/TINAMICA_SPAIN/Laboratorios/Lab_time_series/GitHub_Repositorio/jesus_pernalete/reportes/madrid_polution_diario_001.html')

#Para trabajar las series de tiempo los valores missing los imputamos con 0
madrid_polution_diario_ts=madrid_polution_diario.fillna(0)
prof = ProfileReport(madrid_polution_diario_ts)
prof.to_file(output_file='C:/TINAMICA/TINAMICA_SPAIN/Laboratorios/Lab_time_series/GitHub_Repositorio/jesus_pernalete/reportes/madrid_polution_diario_002.html')


madrid_polution_diario_ts['FECHA'] = madrid_polution_diario_ts.index
madrid_polution_diario_ts.to_csv('C:/TINAMICA/TINAMICA_SPAIN/Laboratorios/Lab_time_series/GitHub_Repositorio/jesus_pernalete/datos/madrid_polution_diario_ts.csv')



madrid_polution_mes = madrid_polution.groupby('YM', as_index=False).agg(    
                                            SEASON=('SEASON','min'), 
                                            
                                            PM_RETIRO= ('PM_RETIRO','mean'),
                                            PM_VALLECAS= ('PM_VALLECAS','mean'),
                                            PM_CIUDADLINEAL= ('PM_CIUDADLINEAL','mean'),
                                            PM_CENTRO= ('PM_CENTRO','mean'),
                                            
                                            DEW_POINT= ('DEW_POINT','mean'),
                                            HUMIDITY= ('HUMIDITY','mean'),
                                            PREASSURE= ('PREASSURE','mean'),
                                            TEMPERATURE= ('TEMPERATURE','mean'),
                                            WIND_SPEED= ('WIND_SPEED','mean'),
                                            COMMULATIVE_PRECIPITATION= ('COMMULATIVE_PRECIPITATION','max'))

madrid_polution_mes_ts=madrid_polution_mes.fillna(0)
madrid_polution_mes_ts['log_PM_CENTRO'] = np.log(madrid_polution_mes_ts.PM_CENTRO) 

prof = ProfileReport(madrid_polution_mes_ts)
prof.to_file(output_file='C:/TINAMICA/TINAMICA_SPAIN/Laboratorios/Lab_time_series/GitHub_Repositorio/jesus_pernalete/reportes/madrid_polution_mes_001.html')


madrid_polution_mes_ts.to_csv('C:/TINAMICA/TINAMICA_SPAIN/Laboratorios/Lab_time_series/GitHub_Repositorio/jesus_pernalete/datos/madrid_polution_mes_ts.csv')

