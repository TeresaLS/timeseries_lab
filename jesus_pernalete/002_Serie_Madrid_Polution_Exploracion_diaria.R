library(plyr)
library(dplyr)
library(lubridate)
library(readxl)
library(forecast)

library(ggplot2)
library(reshape2)

polution_ts <- read.csv("C:/TINAMICA/TINAMICA_SPAIN/Laboratorios/Lab_time_series/GitHub_Repositorio/jesus_pernalete/datos/madrid_polution_diario_ts.csv")
typeof(polution_ts)
colnames(polution_ts)



#sERIE PM_CENTRO

polution_ts_pm_centro<-subset(polution_ts, select=c(PM_CENTRO))
polution_ts_pm_centro<-ts(polution_ts_pm_centro$PM_CENTRO, start = c(2010,1), frequency = 365)
print(polution_ts_pm_centro)


#grafico de la serie
autoplot(polution_ts_pm_centro, ts.colour = "blue", ts.linetype = "dashed")

#grafico componentes de la serie
autoplot(stl(polution_ts_pm_centro, s.window = "periodic"), ts.colour = "blue")

#grafico acf y pacf de la serie

autoplot(acf(polution_ts_pm_centro, plot = FALSE))
pacf(polution_ts_pm_centro)

# Prueba de Raiz Unitaria de Dickey-Fuller# para validar si la serie es o no estacionaria


library (tseries)
q=acf(polution_ts_pm_centro, plot=TRUE)
p=pacf(polution_ts_pm_centro, plot=TRUE)

length(q$lag)
length(p$lag)
adf.test(polution_ts_pm_centro) 


#sERIE temperatura
polution_ts_temperature<-subset(polution_ts, select=c(TEMPERATURE))
polution_ts_temperature<-ts(polution_ts_temperature$TEMPERATURE, start = c(2010,1), frequency = 365)
print(polution_ts_temperature)

#grafico de la serie
autoplot(polution_ts_temperature, ts.colour = "blue", ts.linetype = "dashed")

#grafico componentes de la serie
autoplot(stl(polution_ts_temperature, s.window = "periodic"), ts.colour = "blue")

#grafico acf y pacf de la serie

autoplot(acf(polution_ts_temperature, plot = FALSE))
pacf(polution_ts_temperature)

# Prueba de Raiz Unitaria de Dickey-Fuller# para validar si la serie es o no estacionaria


library (tseries)
q=acf(polution_ts_temperature, plot=TRUE)
p=pacf(polution_ts_temperature, plot=TRUE)

length(q$lag)
length(p$lag)
adf.test(polution_ts_temperature) 
