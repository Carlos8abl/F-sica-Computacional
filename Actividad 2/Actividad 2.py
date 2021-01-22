#!/usr/bin/env python
# coding: utf-8

# # Actividad 2

# In[3]:


import numpy as np


# In[4]:


print(np.pi)


# In[6]:


np.sqrt(np.pi)


# In[7]:


np.sin(np.sqrt(np.pi))


# # Ejercicio 1
En esta parte del ejercicio 1 haremos un programa que nos permita calcular el área de un círculo, para esto usaremos la siguiente ecuación:

$a=\pir^2$ , donde $r$ es el radio de la circunferencia.
# In[19]:


print("Proporcione el radio de la circunferencia: ", end="")
r=float(input())
a=(np.pi*r**2)
print("\nÁrea = ", a)
print("\nÁrea = {:.2f}" .format(a))

Ahora necesitamos hacerlo para una elipse, usaremos la ecuación:

$área=\piab$ ,  donde $a$ y $b$ son el semieje mayor y el semieje menor respectivamente.
# In[20]:


print("Proporcione el semieje mayor y el semieje menor: ", end="")
a=float(input())
b=float(input())
área=(np.pi*a*b)
print("\nÁrea = ", área)
print("\nÁrea = {:.2f}" .format(área))

En este caso nuestro código se encargará de calcular el volumen de una esfera, usaremos la ecuación:

$v=\frac{4}{3}(\pir^3)$
# In[21]:


print("Proporcione el radio de la esfera: ", end="")
r=float(input())
v=(4/3)*(np.pi)*(r**3)
print("\nVolumen = ", v)
print("\nVolumen = {:.2f}" .format(v))

Pára terminar con el ejercicio 1, el siguiente calculará el volumen de un cilindro circular, para ello precisamos de esta ecuación:

$v=\pihr^2$
# In[22]:


print("Proporcione el radio y la altura del cilindro: ", end="")
r=float(input())
h=float(input())
v=np.pi*h*r**2
print("\nVolumen = ", v)
print("\nVolumen = {:.2f}" .format(v))


# # Ejercicio 2

# In[5]:


print("Introduzca los coeficientes de la ecuación cuadrática", end="")
a=float(input())
b=float(input())
c=float(input())

d=b**2-(4*a*c)

if d >= 0:
    
    X=(-b+np.sqrt(d))/2*a

    Y=(-b-np.sqrt(d))/2*a
    print("\nRaices = ", X , Y)
    
    print("\nRaíz 1 = {:.4f}" .format(X))
    print("\nRaíz 2 = {:.4f}" .format(Y))

else:
    print("Las raices de esta ecuación son de números complejos")
    


# # Ejercicio 3

# In[18]:


print("Introduzca el número del que desea saber su raíz cuadráda", end="" )

S=float(input())

X0=np.sqrt(S)

print("La raíz por el método tradicional es:", X0)

print("Introduzca un número arbitrario lo más cercano a", X0)

Xn=float(input())

i=1

XN=1/2*(XN+(S/XN))

while abs(S-XN**2)>0.01:
    
    print("Ciclo", i, ":", XN )
    
    XN=1/2*(XN+(S/XN))
    
    i=i+1
    
    if i>101:
        
        break
        
print("El valor obtenido para la raiz de", S, "por el método de Herón es:", XN)

print("Raíz por el método tradicional:", X0)

print("Raíz por el método de Herón:", XN)

