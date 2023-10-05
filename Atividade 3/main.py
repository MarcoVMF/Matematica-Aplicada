# Matematica Aplicada Usando Conceitos de Runge-Kutta

# Importando bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a = 0
b = 1
n = 10
h = ((b-a)/n)

x = np.arange(a, b+h, h)
ye = np.zeros(n + 1) 
yi = np.zeros(n + 1)
ye[0] = 0
yi[0] = 0

for i in range(n):
    ye[i+1] = ye[i]+h*((1)/(1+(x[i])**2)-2*(ye[i])**2)
    yi[i+1] = (yi[i]+(h)/(1+(x[i+1])**2))/(1+2*h*yi[i])

## Inicializando o vetor t e y

t = np.zeros(n + 1)
y = np.zeros(n + 1)

## Metodo Runge-Kutta

y[0] = 0
t[0] = 0



for i in range(n):
    t1 = t[i]
    y1 = y[i]
    k1 = (1)/(1+(t1)**2)-2*(y1)**2
    t2 = t1+(h/2)
    y2 = y1+(h/2)*k1
    k2 = (1)/(1+(t2)**2)-2*(y2)**2
    y3 = y2+(h/2)*k2
    k3 = (1)/(1+(t2)**2)-2*(y3)**2
    t[i+1] = t[i]+h
    y4 = y1+h*k3
    k4 = (1)/(1+(t[i]+h)**2)-2*(y4)**2
    y[i+1] = y[i]+(h/6)*(k1+2*k2+2*k3+k4)

ext = x/(1+x**2)

## Plotando os graficos

plt.plot(x, ye, 'r', label='Euler Explicito')
plt.plot(x, yi, 'b', label='Euler Implicito')
plt.plot(t, y, 'g', label='Runge-Kutta')
plt.plot(x, ext, 'y', label='Solucao Exata')
plt.legend()
plt.title('Metodo de Aproximacao')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

ea = ext.T - y.T
ea1 = ext.T - ye
ea2 = ext.T - yi
er = (ext.T - y.T)/(ext.T)
er1 = (ext.T - ye.T)/(ext.T)
er2 = (ext.T - yi.T)/(ext.T)

dataRungeKutta = {"X: ": x.T, "Runge-Kutta: ": y.T, "Exata: ": ext.T, "Erro L: ": ea, "Erro R: ": er}
dataAvancada = {"X: ": x.T, "Avancada: ": ye, "Exata: ": ext.T, "Erro L: ": ea1, "Erro R: ": er1}
dataAtrasada = {"X: ": x.T, "Atrasada: ": yi, "Exata: ": ext.T, "Erro L: ": ea2, "Erro R: ": er2}
dataTodos = {"X: ": x.T, "Avancada: ": ye, "Runge-Kutta: ": y.T, "Atrasada: ": yi}

dataFrameRunge = pd.DataFrame(dataRungeKutta)
dataAvancada = pd.DataFrame(dataAvancada)
dataAtrasada = pd.DataFrame(dataAtrasada)
dataTodas = pd.DataFrame(dataTodos)

print(dataFrameRunge)
print(dataAvancada)
print(dataAtrasada)
print(dataTodas)

tabelaRungKutta = pd.DataFrame()