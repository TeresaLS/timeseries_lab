#Vamos a hacer un ejemplo muy simple con los datos de la librería Forecast en R
#Cargamos la librería
library(forecast)



#vamos a visualizar la serie
plot(lynx)


#descomponemos la serie para ver sus factores independientemente
plot(decompose(ts(lynx, frequency=10)))


#para ajustar a un ARIMA vamos a llamar al Auto Arima
forecast::auto.arima(lynx)

#viendo lo que un AR2 MA2 nos daría el mejor resultado automático, vamos a probar hacer un forecast
#creamos el objeto fc con nuestro modelo
mod<-Arima(lynx, order = c(2,0,2), seasonal =list(order=c(1,1,0),period=10))

#vamos a ver cómo se ha ajustado
plot(mod$x, col='black')
lines(fitted(mod),col='red')

#Vamos a visualizar el forecast
prediction=forecast(mod, h=10)
plot(prediction)




















#Hacemos el Dataframe para guardarlo como csv.
lynx_df= as.data.frame(ts(lynx))
# rownames(lynx_df)<-(index= seq(as.Date("1820/01/01"), by = "year", length.out = 114))
# write.csv(lynx_df,'01. Proyectos Ignacio Valenzuela/Formacion_Conocimiento/TimeseriesLab/timeseries_lab/data/Lynx.csv', sep=';')






