# Timeseries lab
Laboratorio de Series Temporales para Tinámica

## Objetivos
    1. Los participantes obtengan conocimiento práctico de análisis y modelado de series temporales con datos de proyectos reales
    2. Los participantes conozcan y apliquen modelos tradicionales y nuevos de series temporales sabiendo evaluarlos y ajustarlos de manera eficiente
    3. Los participantes puedan relizar propuestas y ofertas al cliente con conocimiento práctico de las dimensiones y necesidades de un proyecto de Forecasting. 


## Fase 1
### Creación y Análisis de Series Temporales
   - Creación de Series Temporales
       - Elección fuentes
       - Elección de Resolución temporal
   - Descomposición de Series
       - Tendencia
       - Estacionalidad
       - Ruido
   - Análisis de las Series
       - Tests (ACF & PACF, ADF, etc…)
       - Cross Correlations
   - Transformación de Series Temporales
        - Logaritmos, Diferencias y Normalizaciones
        - Medias Móviles

## Fase 2
### Modelado de Series Temporales
   - Entrenamiento y Validación en Series Temporales
       - Walkforward Validations
       - Métricas de validación (MAE, MAPE, RMSE, MDA)
   - Modelos tradicionales 
      - ARIMAX (SARIMAX), prophet, FFS
      - Creación de variables regresoras, step e impulso
  - Aplicabilidad de los Modelos
      - Teoría de los mercados eficientes 


## Fase 3
### Modelado Avanzado de Series Temporales
   - Modelado de Series Temporales con Redes Neuronales
      - LSTM
   - Simuladores
      - Importancia relativa de variables
      - Identificación de varianza base vs variable




## Vídeos de las sesiones pasadas
Todas las sesiones son grabadas y se pueden ver en el canal de [Youtube](https://www.youtube.com/channel/UCwzfjoGbnFuRMUQk1doOAUw) del laboratorio. 


## Referencias 
Las referencias se encuentran en orden creciente de dificultad. Hay más información basada en R que en Python actualmente, pero el código siempre es bastante replicable.
1. [A Little Book of R For Time Series](https://buildmedia.readthedocs.org/media/pdf/a-little-book-of-r-for-time-series/latest/a-little-book-of-r-for-time-series.pdf) 
Capítulo 2.
2. [Forecasting: Practice and Principle de Rob Hyndman](https://otexts.com/fpp2/)  
3. [Time Series Analysis and Its Applications with R Examples de Shumway y Stoffer](https://www.researchgate.net/publication/265365840_Time_Series_Analysis_and_Its_Applications_With_R_Examples) Este libro es muy técnico. Sirve principalmetne para los que quieran entrar en la matemtica de los modelos de series temporales.

Además, hay vídeos recomendados para varios de los temas que se verán en el laboratorio.
1. [¿Qué es una red LSTM?](https://www.youtube.com/watch?v=1BubAvTVBYs&ab_channel=codificandobits) tiene una explicación muy clara sobre el uso de LSTMs y qué las diferencia de las RNN
2. [Programming LSTMs with Keras and Tensorflow (10.2)](https://www.youtube.com/watch?v=wY0dyFgNCgY&ab_channel=JeffHeaton) de Jeff Heaton entra más en el detalle de cómo implementar el código de keras.

## Paquetes y Software
[RStudio](https://rstudio.com/products/rstudio/) R va a ser útil para algunas secciones del laboratorio, RStudio es el mejor IDE para esto.

[Anaconda](https://www.anaconda.com/products/individual) Con Anaconda podéis descargar Python. Si queréis entrar en el desarrollo de redes neuronales recomiendo que tengáis un enviroment en Python 3.6. 

[Pycharm](https://www.jetbrains.com/es-es/pycharm/download/#section=windows) Mi selección de IDE, os lo recomiendo (Descargad el Community Edition).

[Git Bash](https://git-scm.com/downloads) Para los que no tengan git y estén cómodos utilizando la línea de comandos. 

[Git Extensions](http://gitextensions.github.io/) Versión Open Source de git para los que quieran una interfaz gráfica. 

[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) es la herramienta que utilizaremos para la programación y entrenamiento de redes neuronales.

