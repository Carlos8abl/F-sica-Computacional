#!/usr/bin/env python
# coding: utf-8

# # Actividad 3

# Importamos las librerías

# In[11]:


import numpy as np
import pandas as pd


# In[12]:


url="https://raw.githubusercontent.com/Carlos8abl/F-sica-Computacional-1/main/Actividad%201/Climatolog%C3%ADaDiaria26160.txt"
type_data=['Fecha', 'Precip', 'Evap', 'Tmax', 'Tmin']
df_dist=pd.read_csv(url, names=type_data,encoding='cp1251', sep='\s+', header=None, skiprows=19, skipfooter=1, engine='python')


# Primeras (head) y últimas (tail) 10 filas del archivo.

# In[13]:


print(df_dist.head(10))
print(df_dist.tail(10))


# En la mayor parte de mis datos la precipitación es 0 o nula y la evaporación nula. 

# Dimensiones (shape) del dataframe.

# In[14]:


df_dist.shape


# Información (info) del dataframe.

# In[15]:


df_dist.info()


# Copia (copy) del archivo.

# In[16]:


df_work0=df_dist.copy()


# Primeras 10 filas de la copia.

# In[17]:


df_work0.head(10)


# Reemplazo (replace) del dato "Nulo" por "No medido".

# In[18]:


str_Nulo="Nulo"
df_work1=df_work0.replace(to_replace=str_Nulo, value='No medido', regex=True)


# Primeras 10 filas del archivo con el reemplazo incluido.

# In[19]:


df_work1.head(10)


# Información con los cambios incluidos.

# In[20]:


df_work1.info()
print(df_work1.head(10))


# Convertir a número flotante (numeric) las variables.

# In[21]:


cols_list = ['Precip', 'Evap', 'Tmax', 'Tmin']
for cols in cols_list:
  df_work1[cols] = pd.to_numeric(df_work1[cols], errors='coerce')


# Información actualizada.

# In[22]:


df_work1.info()


# Suma de los valores no medidos.

# In[23]:


df_work1.isnull().sum()


# Primeras y últimas 10 filas del archivo.

# In[24]:


print(df_work1.head(10))
print(df_work1.tail(10))


# Análisis estadístico (describe) con un redondeo (round) de cuatro cifras significativas con el fin de interpretar los datos.

# In[25]:


df_work1.describe().round(4)


# Nueva copia del archivo.

# In[26]:


df_work2=df_work1.copy()


# Primeras 10 filas de la columna Fecha.

# In[27]:


df_work2["Fecha"].head(10)


# Cambio al formato de la fecha (to_datetime) a uno compatible con python.

# In[28]:


df_work2['Fecha']=pd.to_datetime(df_work2['Fecha'], dayfirst=True).copy()


# Información de las varibles (dtypes).

# In[29]:


df_work2.dtypes


# Inclusión de dos columnas adicionales con el año y el mes (dt.year , dt.month)

# In[30]:


df_work2['Año'] = df_work2['Fecha'].dt.year
df_work2['Mes'] = df_work2['Fecha'].dt.month


# Primeras y últimas 10 filas del archvo con las nuevas columnas agregadas. 

# In[31]:


print(df_work2.head(10))
print(df_work2.tail(10))


# Información de todas las varibles incluyendo las nuevas.

# In[32]:


df_work2.info()

