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

polution_ts<- subset(polution_ts, select=-c(FECHA.1))

polution_ts_pm_centro<-subset(polution_ts, select=c(PM_CENTRO))

#allDates <- seq(as.Date('2010-01-01'),as.Date('2015-12-31'),by = "day")
#myts <- ts(polution_ts_pm_centro,
#           start = c(2010, as.numeric(format(allDates[1], "%j"))),
#           frequency = 365)
#print(myts)

#library(ggfortify)
#autoplot(myts, ts.colour = "blue", ts.linetype = "dashed")

#options(repr.pmales.extlot.width=8, repr.plot.height=4)
#plot(myts, main = 'Time Series Polution')

polution_ts_pm_centro<-subset(polution_ts, select=c(PM_CENTRO))
polution_ts_pm_centro<-ts(polution_ts_pm_centro$PM_CENTRO, start = c(2010,1), frequency = 365)
print(polution_ts_pm_centro)

library(ggfortify)
autoplot(polution_ts_pm_centro, ts.colour = "blue", ts.linetype = "dashed")

autoplot(acf(polution_ts_pm_centro, plot = FALSE))
