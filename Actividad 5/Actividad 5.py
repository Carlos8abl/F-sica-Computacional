#!/usr/bin/env python
# coding: utf-8

# # Actividad 5: Análisis de Series de Tiempo

# Librerías que usaremos para esta actividad, en esta actividad se sumará statsmodels.api.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# Características del dataframe anterior.

# In[28]:


url="https://raw.githubusercontent.com/Carlos8abl/F-sica-Computacional-1/main/Actividad%201/Climatolog%C3%ADaDiaria26160.txt"

columnas=["FECHA", "PRECIP", "EVAP", "TMAX", "TMIN"]

df_dist = pd.read_csv(url, names=columnas, encoding='cp1251', sep='\s+', header=None, skiprows=19, skipfooter=1, engine='python')

df_cop0=df_dist.copy()

str_Nulo= "Nulo"
df_cop1 = df_cop0.replace(to_replace=str_Nulo, value='No Medido', regex=True)

cols_list=["PRECIP", "EVAP", "TMAX", "TMIN"]
for cols in cols_list:
  df_cop1[cols]=pd.to_numeric(df_cop1[cols], errors='coerce')

df_cop2=df_cop1.copy()

df_cop2["FECHA"]=pd.to_datetime(df_cop2["FECHA"], dayfirst=True).copy()

df_cop2['Año'] = df_cop2['FECHA'].dt.year
df_cop2['Mes'] = df_cop2['FECHA'].dt.month

df_climat=df_cop2.copy()

df_climat.drop("Mes", axis=1, inplace=True)
df_climat["Mes"] = df_climat["FECHA"].dt.strftime("%b")


# In[29]:


df_climat_ts=df_climat.copy()
df_climat_ts=df_climat_ts.set_index('FECHA')
print(df_climat_ts)


# In[30]:


df_T90=df_climat_ts[(df_climat_ts['Año']>=1990) & (df_climat_ts['Año']<2017)]

Columnas=['PRECIP', 'EVAP', 'TMAX', 'TMIN', 'Año', 'Mes']
df_T90.columns=Columnas

print(df_T90)


# In[33]:


df_T90=df_climat_ts[(df_climat_ts['Año'] >= 1990) & (df_climat_ts['Año'] < 2017)]

sns.set(rc={'figure.figsize':(12, 6)})
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

df_T90['PRECIP'].plot(linewidth=0.5);
plt.ylabel('Precipitación (mm)');
plt.title('Precipitación de los últimos 30 años de Magdalena de Kino', fontsize=18)


# In[34]:


ax1=df_climat_ts.loc['2008','PRECIP'].plot()
ax1.set_ylabel('Precipitación (mm)')
ax1.set_xlabel('Fecha')
plt.title('Precipitación durante 2008 en Magdalena de Kino')
plt.show()


# In[37]:


ax = df_T90.loc['2008-11-01':'2008-12-31', 'PRECIP'].plot(marker='o', linestyle='-')
ax1.set_xlabel('Fecha')
ax.set_ylabel('Precipitación (mm)');
plt.title('Precipitación durante noviembre y diciembre del 2008 en Magdalena de Kino')
plt.show()


# In[38]:


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
plt.figure(figsize=(20,10))
g=sns.barplot(x='Mes', y='PRECIP', data=df_T90);
plt.xlabel('Meses')
plt.ylabel('Precipitación (mm)')
plt.title('Precipitación Promedio por mes y barras de error de los últimos 30 años  para Magdalena de Kino', fontsize=20)
plt.show()


# In[39]:


df_climat_90=df_climat_ts[(df_climat_ts['Año']>=1990)&(df_climat_ts['Año']<2017)]
df_climat_90.columns=Columnas
print(df_climat_90)


# In[43]:


colsplot=['TMAX','TMIN']
ax_ylabels=['Temp. Máxima (°C)', 'Temp. Mínima (°C)','Evaporación (mm)']
Tmax_dT90=df_climat_90['TMAX'].mean()
Tmin_dT90=df_climat_90['TMIN'].mean()
axes=df_climat_90[colsplot].plot(marker='.',alpha=0.4, linestyle='None', figsize=(13,9), subplots=True)

for i, ax in zip(colsplot,axes):
  if i=='TMAX':
    ax.axhline(Tmax_dT90, color='red', linestyle='--', linewidth=4)
    ax.set_ylabel('Temperatura (°C)')
  elif i=='TMIN':
    ax.axhline(Tmin_dT90, color='violet', linestyle='--', linewidth=4)
    ax.set_ylabel('Temperatura (°C)')


# In[46]:


axes=df_climat_90[colsplot].loc["2008"].plot(marker='.',alpha=0.4, linestyle='None', figsize=(13,9), subplots=True)

for i, ax in zip(colsplot,axes):
  if i=='TMAX':
    ax.axhline(Tmax_dT90, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('Temperatura (°C)')
  elif i=='TMIN':
    ax.axhline(Tmin_dT90, color='violet', linestyle='--', linewidth=1)
    ax.set_ylabel('Temperatura (°C)')


# In[62]:


fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
for i, a in zip(['TMAX', 'TMIN'], axes):
    sns.boxplot(data=df_climat_90, x='Mes', y=i, ax=a)
if i=='TMAX':
    a.axhline(Tmax_dT90, color='r', linestyle='--', linewidth=2)
    a.set_ylabel('Temperatura (°C)')
elif i=='TMIN':
    a.axhline(Tmin_dT90, color='m', linestyle='--', linewidth=2)
    a.set_ylabel('Temperatura (°C)')
ax.set_title(i)
if ax != axes[-1]:
    ax.set_xlabel('')


# In[65]:


colosm=["TMAX", "TMIN"]


# In[66]:


df_climat_90_7=df_climat_90[colosm].rolling(7, center=True, min_periods=1).mean()
df_climat_90_30=df_climat_90[colosm].rolling(30, center=True, min_periods=1).mean()
df_climat_90_365=df_climat_90[colosm].rolling(365, center=True, min_periods=1).mean()


# In[67]:


inicio, fin = '2004-01', '2005-01'


# In[71]:


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={'figure.figsize':(20,10)})
fig, ax=plt.subplots()
ax.plot(df_climat_90.loc[inicio:fin, 'TMAX'], marker='.', linestyle='-',color='g', linewidth=0.6, label='Datos diarios')
ax.plot(df_climat_90_7.loc[inicio:fin, 'TMAX'], marker='.', linestyle='--', linewidth=0.6, color='r', label='Promedio móvil de 7 días')
ax.set_xlabel('Fecha')
ax.set_xlabel('Temperatura (°C)')
ax.set_title('Temperaturas máximas con promedio móvil de 7 días en el año 2004 en Magdalena de Kino', fontsize=20)
ax.legend()


# In[72]:


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={'figure.figsize':(20,10)})
fig, ax=plt.subplots()
ax.plot(df_climat_90.loc[inicio:fin, 'TMIN'], marker='.', linestyle='-',color='m', linewidth=0.6, label='Datos diarios')
ax.plot(df_climat_90_7.loc[inicio:fin, 'TMIN'], marker='.', linestyle='--', linewidth=0.6, color='g', label='Promedio móvil de 7 días')
ax.set_xlabel('Fecha')
ax.set_xlabel('Temperatura (°C)')
ax.set_title('Temperaturas mínimas con promedio móvil de 7 días en el año 2004 en Magdalena de Kino', fontsize=20)
ax.legend()


# In[73]:


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={'figure.figsize':(20,10)})
fig, ax=plt.subplots()
ax.plot(df_climat_90.loc[inicio:fin, 'TMAX'], marker='.', linestyle='-',color='g', linewidth=0.6, label='Datos diarios')
ax.plot(df_climat_90_30.loc[inicio:fin, 'TMAX'], marker='.', linestyle='--', linewidth=0.6, color='r', label='Promedio móvil de 30 días')
ax.set_xlabel('Fecha')
ax.set_xlabel('Temperatura (°C)')
ax.set_title('Temperaturas máximas con promedio móvil de 30 días en el año 2004 en Magdalena de Kino', fontsize=20)
ax.legend()


# In[74]:


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={'figure.figsize':(20,10)})
fig, ax=plt.subplots()
ax.plot(df_climat_90.loc[inicio:fin, 'TMIN'], marker='.', linestyle='-',color='m', linewidth=0.6, label='Datos diarios')
ax.plot(df_climat_90_30.loc[inicio:fin, 'TMIN'], marker='.', linestyle='--', linewidth=0.6, color='g', label='Promedio móvil de 30 días')
ax.set_xlabel('Fecha')
ax.set_xlabel('Temperatura (°C)')
ax.set_title('Temperaturas mínimas con promedio móvil de 30 días en el año 2004 en Magdalena de Kino', fontsize=20)
ax.legend()


# In[75]:


inicio, fin = '2004-01', '2016-12'


# In[76]:


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={'figure.figsize':(20,10)})
fig, ax=plt.subplots()
ax.plot(df_climat_90.loc[inicio:fin, 'TMAX'], marker='.', linestyle='-',color='g', linewidth=0.6, label='Datos diarios')
ax.plot(df_climat_90_365.loc[inicio:fin, 'TMAX'], marker='.', linestyle='--', linewidth=0.6, color='r', label='Promedio móvil de 365 días')
ax.set_xlabel('Fecha')
ax.set_xlabel('Temperatura (°C)')
ax.set_title('Temperaturas máximas con promedio móvil de 365 días desde 2004 hasta 2016 en Magdalena de Kino', fontsize=20)
ax.legend()


# In[79]:


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(rc={'figure.figsize':(20,10)})
fig, ax=plt.subplots()
ax.plot(df_climat_90.loc[inicio:fin, 'TMIN'], marker='.', linestyle='-',color='b', linewidth=0.6, label='Datos diarios')
ax.plot(df_climat_90_365.loc[inicio:fin, 'TMIN'], marker='.', linestyle='--', linewidth=0.6, color='m', label='Promedio móvil de 365 días')
ax.set_xlabel('Fecha')
ax.set_xlabel('Temperatura (°C)')
ax.set_title('Temperaturas mínimas con promedio móvil de 365 días desde 2004 hasta 2016 en Magdalena de Kino', fontsize=20)
ax.legend()


# In[84]:


sns.set(rc={'figure.figsize':(16, 8)})
start, end='2010-01', '2016-01'
decompfreq=365
resTMAX = sm.tsa.seasonal_decompose(df_climat_90_365.loc[start:end, 'TMAX'].interpolate(), freq=decompfreq, model='additive')
resTMAX.plot()
plt.title('Descomposición Aditiva de la temperatura máxima (promedio Móvil 365 días)')


# In[86]:


sns.set(rc={'figure.figsize':(16, 8)})
start, end='2010-01', '2016-01'
decompfreq=365
resTMIN = sm.tsa.seasonal_decompose(df_climat_90_365.loc[start:end, 'TMIN'].interpolate(), freq=decompfreq, model='additive')
resTMIN.plot()
plt.title('Descomposición Aditiva de la temperatura mínima (promedio Móvil 365 días)')


# In[87]:


df_T60= df_climat_ts[(df_climat_ts['Año'] >= 1960) & (df_climat_ts['Año'] < 1990)]
df_T90= df_climat_ts[(df_climat_ts['Año'] >= 1990) & (df_climat_ts['Año'] < 2017)]


# In[92]:


g1 = sns.displot(df_T60["TMAX"], kde=True, color='lightcoral', height=4, aspect=2);
plt.axvline(df_T60.TMAX.mean(), linestyle='--', linewidth=2, color='maroon');

g2 = sns.displot(df_T90["TMAX"], kde=True, color='rosybrown', height=4, aspect=2);
plt.axvline(df_T90.TMAX.mean(), linestyle='--', linewidth=2, color='firebrick');

g1.set(title='Temperatura máxima promedio  1960-1989');
g2.set(title='Temperatura máxima promedio 1990-2016');

g1.set(xlim=(10,50))
g2.set(xlim=(10,50))

plt.show();

print('TMAX_promedio df_T60 = ', df_T60.TMAX.mean(), 'ºC' )
print('TMAX_promedio df_T90 = ', df_T90.TMAX.mean(), 'ºC' )


# In[105]:


g1 = sns.displot(df_T60["TMIN"], kde=True, color='cornflowerblue', height=4, aspect=2);
plt.axvline(df_T60.TMIN.mean(), linestyle='--', linewidth=2, color='midnightblue');
g1.set(title='Temperatura mínima promedio 1960-1989');


# In[102]:


g2 = sns.displot(df_T90["TMIN"], kde=True, color='darkslateblue', height=4, aspect=2);
plt.axvline(df_T90.TMIN.mean(), linestyle='--', linewidth=2, color='indigo');
g2.set(title='Temperatura mínima 1990-2016');
g2.set(xlim=(10,50))


# In[99]:


plt.show();

print('TMIN_promedio df_T60 = ', df_T60.TMIN.mean(), 'ºC' )
print('TMIN_promedio df_T90 = ', df_T90.TMIN.mean(), 'ºC' )


# In[ ]:




