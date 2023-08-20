import numpy as np
import pandas as pd

x0 = 5
f = lambda x: x**2
df = lambda x: 2*x

n = 10

espacamento = np.zeros((n, 1))
difAvancada = np.zeros((n, 1))
difAtrasada = np.zeros((n, 1))
difCentrada = np.zeros((n, 1))
erroAvancada = np.zeros((n, 1))
erroAtrasada = np.zeros((n, 1))
erroCentrada = np.zeros((n, 1))

for i in range(n):
    espacamento[i, 0] = 10**(-i)
    difAvancada[i, 0] = (f(x0 + espacamento[i, 0]) - f(x0))/espacamento[i, 0]
    difAtrasada[i, 0] = (f(x0) - f(x0 - espacamento[i, 0]))/espacamento[i, 0]
    difCentrada[i, 0] = (f(x0 + espacamento[i, 0]) - f(x0 - espacamento[i, 0]))/(2*espacamento[i, 0])
    erroAvancada[i, 0] = np.abs((df(x0) - difAvancada[i, 0])/df(x0))
    erroAtrasada[i, 0] = np.abs((df(x0) - difAtrasada[i, 0])/df(x0))
    erroCentrada[i, 0] = np.abs((df(x0) - difCentrada[i, 0])/df(x0))

dataAvancada = {'Espaçamento:': espacamento.flatten(), 'Avançada:': difAvancada.flatten(), 'Erro:': erroAvancada.flatten()}
dataAtrasada = {'Espaçamento:': espacamento.flatten(), 'Atrasada:': difAtrasada.flatten(), 'Erro:': erroAtrasada.flatten()}
dataCentrada = {'Espaçamento:': espacamento.flatten(), 'Centrada:': difCentrada.flatten(), 'Erro:': erroCentrada.flatten()}

dataFrameAvancada = pd.DataFrame(dataAvancada)
dataFrameAtrasada = pd.DataFrame(dataAtrasada)
dataFrameCentrada = pd.DataFrame(dataCentrada)

print("Avançada:")
print(dataFrameAvancada)

print("Atrasada:")
print(dataFrameAtrasada)

print("Centrada:")
print(dataFrameCentrada)
