#!/usr/bin/env python
# coding: utf-8

# # Actividad 6: Pronóstico de series de tiempo

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.tsa.stattools import acf, pacf
import pandas.util.testing as tm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


url="https://raw.githubusercontent.com/Carlos8abl/F-sica-Computacional-1/main/Actividad%201/Climatolog%C3%ADaDiaria26160.txt"
my_cols = ['Fecha', 'Precip', 'Evap', 'Tmax', 'Tmin']
df_dist = pd.read_csv(url, names=my_cols, encoding='cp1251', sep='\s+', header=None, skiprows=19, skipfooter=1, engine='python')
df_work0 = df_dist.copy()
str_Nulo = 'Nulo'
df_work1 = df_work0.replace(to_replace=str_Nulo, value='', regex=True)
cols_list = ['Precip', 'Evap', 'Tmax', 'Tmin']
for cols in cols_list:
  df_work1[cols] = pd.to_numeric(df_work1[cols], errors='coerce')
df_work2 = df_work1.copy()
df_work2['Fecha']=pd.to_datetime(df_work2['Fecha'], dayfirst=True).copy()
df_work2['Año'] = df_work2['Fecha'].dt.year
df_work2['Mes'] = df_work2['Fecha'].dt.strftime('%b')
df_meteo = df_work2.copy()
df_meteo_ts = df_meteo.copy()
df_meteo_ts = df_meteo_ts.set_index('Fecha')
df_meteo_ts.info()
df_30 = df_meteo_ts[(df_meteo_ts['Año'] >= 1930) & (df_meteo_ts['Año'] < 1960)]
df_60 = df_meteo_ts[(df_meteo_ts['Año'] >= 1960) & (df_meteo_ts['Año'] < 1990)]
df_90 = df_meteo_ts[(df_meteo_ts['Año'] >= 1990) & (df_meteo_ts['Año'] < 2018)]


# In[3]:


def test_stationarity(df_ts):
    
    rolmean = df_ts.rolling(window=365, center=True).mean()
    rolstd = df_ts.rolling(window=365, center=True).std() 
    
    plt.rcParams["figure.figsize"] = (12,6)
    
    plt.plot(df_ts, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std Deviation')
 
    plt.legend(loc='best')
    plt.title('Promedio Móvil y Desviación Estándar')
    plt.grid()
    plt.show()
 
    print('Resultados de la Prueba de Dickey-Fuller:')
    dftest = adfuller(df_ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# In[8]:


start, end = '1994-01', '1998-01'

ts_90 = df_90.loc[start:end].copy()
ts_90.head()


# In[9]:


ts_90['Tmax'].head()


# In[10]:


ts_test = ts_90['Tmax'].copy()


# In[11]:


test_stationarity(ts_test)


# In[12]:


ts_test_log = np.log10(ts_test)
plt.plot(ts_test_log)
plt.xlabel('Años',color='b')
plt.ylabel('Escala Log',color='b')
plt.title('Gráfica utilizando Escala Log(10)')
plt.grid()
plt.show()


# In[13]:


moving_avg = ts_test_log.rolling(365, center=True).mean()
plt.plot(ts_test_log)
plt.plot(moving_avg, color = 'brown')
plt.xlabel('Años',color='b')
plt.ylabel('Escala Log',color='b')
plt.title('Valores y Promedio Móvil')
plt.grid()
plt.show()


# In[14]:


ts_test_log_moving_ave_diff = ts_test_log - moving_avg 
ts_test_log_moving_ave_diff.head()


# In[15]:


ts_test_log_moving_ave_diff.dropna(inplace=True)
ts_test_log_moving_ave_diff.head()


# In[16]:


test_stationarity(ts_test_log_moving_ave_diff)


# In[17]:


EWM_avg = ts_test_log.ewm(halflife=30).mean()
plt.plot(ts_test_log)
plt.plot(EWM_avg, color = 'brown')
plt.xlabel('Años',color='b')
plt.ylabel('Escala Log',color='b')
plt.title('Promedio Móvil Exponencial ')
plt.grid()
plt.show()


# In[18]:


EWM_avg.head()


# In[19]:


ts_test_log_EWM_avg_diff = ts_test_log - EWM_avg


# In[20]:


test_stationarity(ts_test_log_EWM_avg_diff)


# In[22]:


ts_test_log_diff = ts_test_log - ts_test_log.shift(periods=7)
plt.plot(ts_test_log_diff)


# In[23]:


ts_test_log_diff.dropna(inplace=True)
test_stationarity(ts_test_log_diff)


# In[24]:


len(ts_test_log_diff)


# In[25]:


decomposition = sm.tsa.seasonal_decompose(ts_test_log.interpolate(), model='additive', freq=365)

trend = decomposition.trend
seasonal = decomposition.seasonal
residue = decomposition.resid

plt.rcParams["figure.figsize"] = (12,12)
plt.subplot(411)
plt.plot(ts_test_log, label='Observado (Escala Log)')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Tendencia')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Estacionalidad')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residue, label='Residuo')
plt.legend(loc='best')
plt.tight_layout()


# In[26]:


ts_test_log_decompose = residue 
ts_test_log_decompose.dropna(inplace=True)
test_stationarity(ts_test_log_decompose)


# In[27]:


lag_acf = acf(ts_test_log_diff, nlags=20)
lag_pacf = pacf(ts_test_log_diff, nlags=20, method = 'ols')

plt.rcParams["figure.figsize"] = (12,6)

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_test_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_test_log_diff)), linestyle='--', color='green')

plt.axvline(x=4, linestyle='--', color='green')
plt.grid()
plt.title('Autocorrelation Function (Ubica el valor de q)')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_test_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_test_log_diff)), linestyle='--', color='green')

plt.axvline(x=2, linestyle='--', color='green')
plt.grid()
plt.title('Partial Autocorrelation Function (Ubica el valor de p)')

plt.tight_layout()


# In[28]:


model = ARIMA(ts_test_log, order=(2,1,0), missing='drop')
results_AR = model.fit(displ=-1)
plt.plot(ts_test_log_diff)
plt.plot(results_AR.fittedvalues, color='brown')
plt.title('ARIMA(2,1,0) = AR(2)')
print(results_AR.summary())


# In[29]:


model = ARIMA(ts_test_log, order=(0,1,4), missing='drop')
results_MA = model.fit(displ=-1)
plt.plot(ts_test_log_diff)
plt.plot(results_MA.fittedvalues, color='brown')
plt.title('ARIMA(0,1,4) = MA(4)')
print(results_MA.summary())


# In[30]:


model = ARIMA(ts_test_log, order=(2,1,4), missing='drop')
results_ARIMA = model.fit(displ=-1)
plt.plot(ts_test_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='brown')
plt.title('ARIMA(2,1,4) = AR(2) + MA(4)')
print(results_ARIMA.summary())


# In[31]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff)


# In[32]:


predictions_ARIMA_log = pd.Series(ts_test_log, index = ts_test_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff)
print(predictions_ARIMA_log.head())
print(predictions_ARIMA_log.tail())


# In[33]:


predictions_ARIMA = 10**(predictions_ARIMA_log)

plt.rcParams["figure.figsize"] = (12,6)

plt.plot(ts_test, label='Observado');
plt.plot(predictions_ARIMA, label='Modelo ARIMA');
plt.xlabel('Fecha')
plt.ylabel('Tmax (ºC)')
plt.title('La serie observada y la predicción del modelo ARIMA para Tmax')
plt.legend(loc='best')


# In[34]:


RMSE = np.sqrt(np.mean((predictions_ARIMA - ts_test)**2))
print('RMSE:', RMSE)


# In[35]:


ts_test_last = ts_test.tail(730)
predictions_ARIMA_last = predictions_ARIMA.tail(730)
plt.plot(ts_test_last, label='Observada');
plt.plot(predictions_ARIMA_last, label='Modelo ARIMA');
plt.xlabel('Fecha')
plt.ylabel('Tmax (ºC)')
plt.title('La serie observada y la predicción del modelo ARIMA')
plt.legend(loc='best');


# In[52]:


start, end = '1995-01', '1999-01'

ts_90 = df_90.loc[start:end].copy()
ts_90.head()


# In[53]:


ts_test2 = ts_90['Tmin'].copy()


# In[54]:


test_stationarity(ts_test2)


# In[59]:


ts_test2_cbrt = np.cbrt(ts_test2)
plt.plot(ts_test2_cbrt)
plt.xlabel('Años',color='b')
plt.ylabel('Escala Raíz Cúbica',color='b')
plt.title('Gráfica utilizando Escala Raíz Cúbica')
plt.grid()
plt.show()


# In[60]:


moving2_avg = ts_test2_cbrt.rolling(365, center=True).mean()
plt.plot(ts_test2_cbrt)
plt.plot(moving_avg, color = 'brown')
plt.xlabel('Años',color='b')
plt.ylabel('Escala Raíz Cúbica',color='b')
plt.title('Valores y Promedio Móvil')
plt.grid()
plt.show()


# In[61]:


ts_test2_cbrt_moving_ave_diff = ts_test2_cbrt - moving_avg 
ts_test2_cbrt_moving_ave_diff.head()


# In[62]:


ts_test2_cbrt_moving_ave_diff.dropna(inplace=True)
ts_test2_cbrt_moving_ave_diff.head()


# In[63]:


test_stationarity(ts_test2_cbrt_moving_ave_diff)


# In[64]:


EWM2_avg = ts_test2_cbrt.ewm(halflife=30).mean()
plt.plot(ts_test2_cbrt)
plt.plot(EWM2_avg, color = 'brown')
plt.xlabel('Años',color='b')
plt.ylabel('Escala Raíz Cúbica',color='b')
plt.title('Promedio Móvil Exponencial ')
plt.grid()
plt.show()


# In[70]:


ts_test2_cbrt_EWM2_avg_diff = ts_test2_cbrt - EWM_avg
EWM2_avg.head()


# In[71]:


ts_test2_cbrt_EWM2_avg_diff.dropna(inplace=True)


# In[72]:


test_stationarity(ts_test2_cbrt_EWM2_avg_diff)


# In[73]:


ts_test2_cbrt_diff = ts_test2_cbrt - ts_test2_cbrt.shift(periods=7)
plt.plot(ts_test2_cbrt_diff)


# In[74]:


ts_test2_cbrt_diff.dropna(inplace=True)
test_stationarity(ts_test2_cbrt_diff)


# In[76]:


des = sm.tsa.seasonal_decompose(ts_test2_cbrt.interpolate(), model='additive', period=365)


trend = des.trend
seasonal = des.seasonal
residue = des.resid

plt.rcParams["figure.figsize"] = (12,12)
plt.subplot(411)
plt.plot(ts_test2_cbrt, label='Observado (Escala Raíz Cúbica)')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Tendencia')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Estacionalidad')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residue, label='Residuo')
plt.legend(loc='best')

plt.tight_layout()


# In[77]:


ts_test2_cbrt_decompose = residue 
ts_test2_cbrt_decompose.dropna(inplace=True)
test_stationarity(ts_test2_cbrt_decompose)


# In[78]:


lag_acf = acf(ts_test2_cbrt_diff, nlags=20)
lag_pacf = pacf(ts_test2_cbrt_diff, nlags=20, method = 'ols')

plt.rcParams["figure.figsize"] = (12,6)

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_test2_cbrt_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_test2_cbrt_diff)), linestyle='--', color='green')

plt.axvline(x=4, linestyle='--', color='green')
plt.grid()
plt.title('Autocorrelation Function (Ubica el valor de q)')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_test2_cbrt_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_test2_cbrt_diff)), linestyle='--', color='green')

plt.axvline(x=2, linestyle='--', color='green')
plt.grid()
plt.title('Partial Autocorrelation Function (Ubica el valor de p)')

plt.tight_layout()


# In[79]:


model = ARIMA(ts_test2_cbrt, order=(2,1,0), missing='drop')
results_AR2 = model.fit(displ=-1)
plt.plot(ts_test2_cbrt_diff)
plt.plot(results_AR2.fittedvalues, color='brown')
plt.title('ARIMA(2,1,0) = AR(2)')
print(results_AR2.summary())


# In[80]:


model = ARIMA(ts_test2_cbrt, order=(0,1,4), missing='drop')
results_MA2 = model.fit(displ=-1)
plt.plot(ts_test2_cbrt_diff)
plt.plot(results_MA2.fittedvalues, color='brown')
plt.title('ARIMA(0,1,4) = MA(4)')
print(results_MA2.summary())


# In[81]:


model = ARIMA(ts_test2_cbrt, order=(2,1,4), missing='drop')
results_ARIMA2 = model.fit(displ=-1)
plt.plot(ts_test2_cbrt_diff)
plt.plot(results_ARIMA2.fittedvalues, color='brown')
plt.title('ARIMA(2,1,4) = AR(2) + MA(4)')
print(results_ARIMA2.summary())


# In[82]:


predictions2_ARIMA_diff = pd.Series(results_ARIMA2.fittedvalues, copy=True)
print(predictions2_ARIMA_diff)


# In[83]:


predictions2_ARIMA_cbrt = pd.Series(ts_test2_cbrt, index = ts_test2_cbrt.index)
predictions2_ARIMA_cbrt = predictions2_ARIMA_cbrt.add(predictions2_ARIMA_diff)
print(predictions2_ARIMA_cbrt.head())
print(predictions2_ARIMA_cbrt.tail())


# In[107]:


predictions2_ARIMA = 8*(predictions2_ARIMA_cbrt)

plt.rcParams["figure.figsize"] = (12,6)

plt.plot(ts_test2, label='Observado');
plt.plot(predictions2_ARIMA, label='Modelo ARIMA');
plt.xlabel('Fecha')
plt.ylabel('Tmin (ºC)')
plt.title('La serie observada y la predicción del modelo ARIMA para Tmin')
plt.legend(loc='best')


# In[108]:


RMSE = np.sqrt(np.mean((predictions2_ARIMA - ts_test2)))
print('RMSE:', RMSE)


# In[109]:


ts_test2_last = ts_test2.tail(1200)
predictions2_ARIMA_last = predictions2_ARIMA.tail(1200)
plt.plot(ts_test2_last, label='Observada');
plt.plot(predictions2_ARIMA_last, label='Modelo ARIMA');
plt.xlabel('Fecha')
plt.ylabel('Tmin (ºC)')
plt.title('La serie observada y la predicción del modelo ARIMA')
plt.legend(loc='best')

