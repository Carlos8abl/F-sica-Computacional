#!/usr/bin/env python
# coding: utf-8

# # Actividad 2

# In[8]:


import numpy as np


# In[9]:


import matplotlib.pyplot as plt


# # Ejemplos de clase

# In[11]:


print(np.pi)


# In[6]:


np.sqrt(np.pi)


# In[15]:


np.sin(np.sqrt(np.pi))


# In[16]:


lista1 = [1,2,3,4]
lista2 = [1,2,3,4]

print(lista1)
print(lista2)
print(lista1+lista2)


# In[17]:


s1 = np.array([1,2,3,4])      
s2 = np.array([1,2,3,4.0])   
s3 = np.array([1,2,3,'4'])     
s4 = np.arange(0,5,0.5)     
s5 = np.linspace(0,5,20) 

print(s1[2])      
print(s1[1:3])

x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

x1 = np.array([[0, 1], [1, 5]])
y1 = np.array([[4, 0], [0, 4]])

x+y     
x/y  
2*x    
x**2     
x @ y             
np.dot(x,y)      
np.sum(x)     
np.sqrt(x)      
np.exp(x)       
np.log(x)


# In[18]:


a = np.array([[1,2,3],
              [6,9,3],
              [5,7,3],
              [4,5,1]])
print(a)


# In[19]:


a[:,1]


# In[20]:


x = np.linspace(-np.pi,np.pi,100)
y = np.sin(x)     
y1 = x
y3 = x**3
y5 = x**5
sT = y1 - y3 + y5 

plt.subplot(111)
plt.xlabel("X label")
plt.ylabel("Y label")
plt.title("Title")

plt.plot(x,y, label="sin(x)")
plt.plot(x,y1, label='x')
plt.plot(x,sT, label='Serie Taylor Orden 5')

plt.grid(True)
plt.ylim(-2, 2)
plt.legend(bbox_to_anchor=(1,1), loc="upper left")
plt.show()


# # Ejercicio 1
En esta parte del ejercicio 1 haremos un programa que nos permita calcular el área de un círculo, para esto usaremos la siguiente ecuación:

$a=\pi r^2$ , donde $r$ es el radio de la circunferencia.
# In[19]:


print("Proporcione el radio de la circunferencia: ", end="")
r=float(input())
a=(np.pi*r**2)
print("\nÁrea = ", a)
print("\nÁrea = {:.2f}" .format(a))

Ahora necesitamos hacerlo para una elipse, usaremos la ecuación:

$area=\pi ab$ ,  donde $a$ y $b$ son el semieje mayor y el semieje menor respectivamente.
# In[20]:


print("Proporcione el semieje mayor y el semieje menor: ", end="")
a=float(input())
b=float(input())
área=(np.pi*a*b)
print("\nÁrea = ", área)
print("\nÁrea = {:.2f}" .format(área))

En este caso nuestro código se encargará de calcular el volumen de una esfera, usaremos la ecuación:

$v=\frac{4}{3}(\pi r^3)$
# In[21]:


print("Proporcione el radio de la esfera: ", end="")
r=float(input())
v=(4/3)*(np.pi)*(r**3)
print("\nVolumen = ", v)
print("\nVolumen = {:.2f}" .format(v))

Pára terminar con el ejercicio 1, el siguiente calculará el volumen de un cilindro circular, para ello precisamos de esta ecuación:

$v=\pi hr^2$
# In[22]:


print("Proporcione el radio y la altura del cilindro: ", end="")
r=float(input())
h=float(input())
v=np.pi*h*r**2
print("\nVolumen = ", v)
print("\nVolumen = {:.2f}" .format(v))


# # Ejercicio 2

# In[10]:


print("Introduzca los coeficientes de la ecuación cuadrática", end="")
a=float(input())
b=float(input())
c=float(input())

d=b**2-(4*a*c)

if d >= 0:
    
    X=(-b+np.sqrt(d))/(2*a)

    Y=(-b-np.sqrt(d))/(2*a)
    print("\nRaices = ", X , Y)
    
    print("\nRaíz 1 = {:.4f}" .format(X))
    print("\nRaíz 2 = {:.4f}" .format(Y))
    
elif d==0:
    
    X=(-b+np.sqrt(d))/(2*a)
    
    Y=(-b+np.sqrt(d))/(2*a)
    
    print("\nRaices =", X , Y)
    
    print("\nRaíz 1 = {:.4f}" .format(X))
    
    print("\nRaíz 2 = {:.4f}" .format(Y))
    

else:
    print("Las raices de esta ecuación son de números complejos")
    


# # Ejercicio 3

# In[13]:


print("Introduzca el número del que desea saber su raíz cuadráda", end="" )

S=float(input())

X0=np.sqrt(S)

print("La raíz por el método tradicional es:", X0)

print("Introduzca un número arbitrario lo más cercano a", X0)

XN=float(input())

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

print("Error en la aproximación de:", (S-XN**2))


# # Ejercicio 4

# In[23]:


x=np.linspace(-1.5,1.5,100) 
y=np.log(1+x)

T4=x-x**2/2+x**3/3-x**4/4
T7=x-x**2/2+x**3/3-x**4/4+x**5/5-x**6/6+x**7/7
T11=(x-x**2/2+x**3/3-x**4/4+x**5/5-x**6/6+x**7/7-x**8/8+x**9/9-x**(10)/10+x**(11)/11)
T16=(x-x**2/2+x**3/3-x**4/4+x**5/5-x**6/6+x**7/7-x**8/8+x**9/9-x**(10)/10+x**(11)/11-x**(12)/12+x**(13)/13-x**(14)/14+x**(15)/15-x**(16)/16)

plt.subplot(111)
plt.xlabel("X")
plt.ylabel("Y")

plt.plot(x,sT4, label='Serie de Taylor de orden 4')
plt.plot(x,sT7, label='Serie de Taylor de orden 7')
plt.plot(x,sT11, label='Serie de Taylor de orden 11')
plt.plot(x,sT16, label='Serie de Taylor de orden 16')
plt.plot(x,y, label='ln(1+x)')

plt.grid(True)

plt.ylim(-4, 2)

plt.legend(bbox_to_anchor=(1,1), loc="upper left")

plt.show()


# In[ ]:




