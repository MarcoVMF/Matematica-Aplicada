import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def exercicioA():
    print("Exercicio A ===============================================================================================================================================================")

    f = lambda x: np.sin(np.exp(2 * x))
    df = lambda x: 2 * (np.exp(2 * x)) * np.cos(np.exp(2 * x))
    inicio = 0
    fim = 1
    n = 40
    h = (fim - inicio)/n

    x = np.arange(inicio, fim+h, h)
    x = x.reshape(-1, 1)

    difAtrasada = (f(x + h) - f(x)) / h
    difAvancada = (f(x) - f(x - h)) / h
    difCentrada = (f(x + h) - f(x - h)) / (2 * h)
    solucaoAnalitica = df(x)

    erroLocalAtrasada = np.abs(solucaoAnalitica - difAtrasada)
    erroLocalAvancada = np.abs(solucaoAnalitica - difAvancada)
    erroLocalCentrada = np.abs(solucaoAnalitica - difCentrada)

    erroRelativoAtrasada = np.abs((solucaoAnalitica - difAtrasada) / solucaoAnalitica)
    erroRelativoAvancada = np.abs((solucaoAnalitica - difAvancada) / solucaoAnalitica)
    erroRelativoCentrada = np.abs((solucaoAnalitica - difCentrada) / solucaoAnalitica)

    dataAtrasada = {'Atrasada:': difAtrasada.flatten(), 'Erro Local:': erroLocalAtrasada.flatten(), 'Erro Relativo:': erroRelativoAtrasada.flatten()}
    dataAvancada = {'Avancada:': difAvancada.flatten(), 'Erro Local:': erroLocalAvancada.flatten(), 'Erro Relativo:': erroRelativoAvancada.flatten()}
    dataCentrada = {'Centrada:': difCentrada.flatten(), 'Erro Local:': erroLocalCentrada.flatten(), 'Erro Relativo:': erroRelativoCentrada.flatten()}

    dataFrameAtrasada = pd.DataFrame(dataAtrasada)
    dataFrameAvancada = pd.DataFrame(dataAvancada)
    dataFrameCentrada = pd.DataFrame(dataCentrada)

    print("Atrasada: ")
    print(dataFrameAtrasada)
    print("Avancada: ")
    print(dataFrameAvancada)
    print("Centrada: ")
    print(dataFrameCentrada)

    plt.plot(x, difAtrasada, 'r-o', label='Dif. Atrasada')
    plt.plot(x, difAvancada, 'b-*', label='Dif. Avançada')
    plt.plot(x, difCentrada, 'k--^', label='Dif. Centrada')
    plt.plot(x, solucaoAnalitica, 'm-*', label='Solução Analítica')
    plt.legend()
    plt.title('Exercício A - Gráficos dos Métodos de Diferenças')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def exercicioB():
    print("Exercicio B ===============================================================================================================================================================")

    f = lambda x: np.sin(x) / np.log(x)
    df = lambda x: ((np.cos(x) * np.log(x)) - (np.sin(x) / x)) / (np.log(x)**2)

    inicio = 6
    fim = 7
    n = 20
    h = (fim - inicio)/n

    x = np.arange(inicio, fim+h, h)
    x = x.reshape(-1, 1)

    difAtrasada = (f(x + h) - f(x)) / h
    difAvancada = (f(x) - f(x - h)) / h
    difCentrada = (f(x + h) - f(x - h)) / (2 * h)
    solucaoAnalitica = df(x)

    erroLocalAtrasada = np.abs(solucaoAnalitica - difAtrasada)
    erroLocalAvancada = np.abs(solucaoAnalitica - difAvancada)
    erroLocalCentrada = np.abs(solucaoAnalitica - difCentrada)

    erroRelativoAtrasada = np.abs((solucaoAnalitica - difAtrasada) / solucaoAnalitica)
    erroRelativoAvancada = np.abs((solucaoAnalitica - difAvancada) / solucaoAnalitica)
    erroRelativoCentrada = np.abs((solucaoAnalitica - difCentrada) / solucaoAnalitica)

    dataAtrasada = {'Atrasada:': difAtrasada.flatten(), 'Erro Local:': erroLocalAtrasada.flatten(), 'Erro Relativo:': erroRelativoAtrasada.flatten()}
    dataAvancada = {'Avancada:': difAvancada.flatten(), 'Erro Local:': erroLocalAvancada.flatten(), 'Erro Relativo:': erroRelativoAvancada.flatten()}
    dataCentrada = {'Centrada:': difCentrada.flatten(), 'Erro Local:': erroLocalCentrada.flatten(), 'Erro Relativo:': erroRelativoCentrada.flatten()}

    dataFrameAtrasada = pd.DataFrame(dataAtrasada)
    dataFrameAvancada = pd.DataFrame(dataAvancada)
    dataFrameCentrada = pd.DataFrame(dataCentrada)

    print("Atrasada: ")
    print(dataFrameAtrasada)
    print("Avancada: ")
    print(dataFrameAvancada)
    print("Centrada: ")
    print(dataFrameCentrada)

    plt.plot(x, difAtrasada, 'r-o', label='Dif. Atrasada')
    plt.plot(x, difAvancada, 'b-*', label='Dif. Avançada')
    plt.plot(x, difCentrada, 'k--^', label='Dif. Centrada')
    plt.plot(x, solucaoAnalitica, 'm-*', label='Solução Analítica')
    plt.legend()
    plt.title('Exercício B - Gráficos dos Métodos de Diferenças')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def exercicioC():
    print("Exercicio C ===============================================================================================================================================================")

    f = lambda x: np.log(x) * np.sin(x)
    df = lambda x: (np.sin(x) / x) + (np.log(x) * np.cos(x))
    inicio = 0.5
    fim = 1.5
    n = 20
    h = (fim - inicio)/n

    x = np.arange(inicio, fim+h, h)
    x = x.reshape(-1, 1)

    difAtrasada = (f(x + h) - f(x)) / h
    difAvancada = (f(x) - f(x - h)) / h
    difCentrada = (f(x + h) - f(x - h)) / (2 * h)
    solucaoAnalitica = df(x)

    erroLocalAtrasada = np.abs(solucaoAnalitica - difAtrasada)
    erroLocalAvancada = np.abs(solucaoAnalitica - difAvancada)
    erroLocalCentrada = np.abs(solucaoAnalitica - difCentrada)

    erroRelativoAtrasada = np.abs((solucaoAnalitica - difAtrasada) / solucaoAnalitica)
    erroRelativoAvancada = np.abs((solucaoAnalitica - difAvancada) / solucaoAnalitica)
    erroRelativoCentrada = np.abs((solucaoAnalitica - difCentrada) / solucaoAnalitica)

    dataAtrasada = {'Atrasada:': difAtrasada.flatten(), 'Erro Local:': erroLocalAtrasada.flatten(), 'Erro Relativo:': erroRelativoAtrasada.flatten()}
    dataAvancada = {'Avancada:': difAvancada.flatten(), 'Erro Local:': erroLocalAvancada.flatten(), 'Erro Relativo:': erroRelativoAvancada.flatten()}
    dataCentrada = {'Centrada:': difCentrada.flatten(), 'Erro Local:': erroLocalCentrada.flatten(), 'Erro Relativo:': erroRelativoCentrada.flatten()}

    dataFrameAtrasada = pd.DataFrame(dataAtrasada)
    dataFrameAvancada = pd.DataFrame(dataAvancada)
    dataFrameCentrada = pd.DataFrame(dataCentrada)

    print("Atrasada: ")
    print(dataFrameAtrasada)
    print("Avancada: ")
    print(dataFrameAvancada)
    print("Centrada: ")
    print(dataFrameCentrada)

    plt.plot(x, difAtrasada, 'r-o', label='Dif. Atrasada')
    plt.plot(x, difAvancada, 'b-*', label='Dif. Avançada')
    plt.plot(x, difCentrada, 'k--^', label='Dif. Centrada')
    plt.plot(x, solucaoAnalitica, 'm-*', label='Solução Analítica')
    plt.legend()
    plt.title('Exercício C - Gráficos dos Métodos de Diferenças')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def exercicioD():
    print("Exercicio D ===============================================================================================================================================================")

    f = lambda x: np.exp(np.sin(x))
    df = lambda x: np.cos(x) * np.exp(np.sin(x))
    inicio = 0.5
    fim = 1.5
    n = 20
    h = (fim - inicio)/n

    x = np.arange(inicio, fim+h, h)
    x = x.reshape(-1, 1)

    difAtrasada = (f(x + h) - f(x)) / h
    difAvancada = (f(x) - f(x - h)) / h
    difCentrada = (f(x + h) - f(x - h)) / (2 * h)
    solucaoAnalitica = df(x)

    erroLocalAtrasada = np.abs(solucaoAnalitica - difAtrasada)
    erroLocalAvancada = np.abs(solucaoAnalitica - difAvancada)
    erroLocalCentrada = np.abs(solucaoAnalitica - difCentrada)

    erroRelativoAtrasada = np.abs((solucaoAnalitica - difAtrasada) / solucaoAnalitica)
    erroRelativoAvancada = np.abs((solucaoAnalitica - difAvancada) / solucaoAnalitica)
    erroRelativoCentrada = np.abs((solucaoAnalitica - difCentrada) / solucaoAnalitica)

    dataAtrasada = {'Atrasada:': difAtrasada.flatten(), 'Erro Local:': erroLocalAtrasada.flatten(), 'Erro Relativo:': erroRelativoAtrasada.flatten()}
    dataAvancada = {'Avancada:': difAvancada.flatten(), 'Erro Local:': erroLocalAvancada.flatten(), 'Erro Relativo:': erroRelativoAvancada.flatten()}
    dataCentrada = {'Centrada:': difCentrada.flatten(), 'Erro Local:': erroLocalCentrada.flatten(), 'Erro Relativo:': erroRelativoCentrada.flatten()}

    dataFrameAtrasada = pd.DataFrame(dataAtrasada)
    dataFrameAvancada = pd.DataFrame(dataAvancada)
    dataFrameCentrada = pd.DataFrame(dataCentrada)

    print("Atrasada: ")
    print(dataFrameAtrasada)
    print("Avancada: ")
    print(dataFrameAvancada)
    print("Centrada: ")
    print(dataFrameCentrada)

    plt.plot(x, difAtrasada, 'r-o', label='Dif. Atrasada')
    plt.plot(x, difAvancada, 'b-*', label='Dif. Avançada')
    plt.plot(x, difCentrada, 'k--^', label='Dif. Centrada')
    plt.plot(x, solucaoAnalitica, 'm-*', label='Solução Analítica')
    plt.legend()
    plt.title('Exercício D - Gráficos dos Métodos de Diferenças')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def exercicioE():
    print("Exercicio E ===============================================================================================================================================================")

    f = lambda x: 5 * (x**4) + 4 * (x**3) + 3 * (x**2) + 2 * x
    df = lambda x: 20*(x**3) + 12*(x**2) + 6*x + 2
    inicio = 0
    fim = 1
    n = 20
    h = (fim - inicio)/n

    x = np.arange(inicio, fim+h, h)
    x = x.reshape(-1, 1)

    difAtrasada = (f(x + h) - f(x)) / h
    difAvancada = (f(x) - f(x - h)) / h
    difCentrada = (f(x + h) - f(x - h)) / (2 * h)
    solucaoAnalitica = df(x)

    erroLocalAtrasada = np.abs(solucaoAnalitica - difAtrasada)
    erroLocalAvancada = np.abs(solucaoAnalitica - difAvancada)
    erroLocalCentrada = np.abs(solucaoAnalitica - difCentrada)

    erroRelativoAtrasada = np.abs((solucaoAnalitica - difAtrasada) / solucaoAnalitica)
    erroRelativoAvancada = np.abs((solucaoAnalitica - difAvancada) / solucaoAnalitica)
    erroRelativoCentrada = np.abs((solucaoAnalitica - difCentrada) / solucaoAnalitica)

    dataAtrasada = {'Atrasada:': difAtrasada.flatten(), 'Erro Local:': erroLocalAtrasada.flatten(), 'Erro Relativo:': erroRelativoAtrasada.flatten()}
    dataAvancada = {'Avancada:': difAvancada.flatten(), 'Erro Local:': erroLocalAvancada.flatten(), 'Erro Relativo:': erroRelativoAvancada.flatten()}
    dataCentrada = {'Centrada:': difCentrada.flatten(), 'Erro Local:': erroLocalCentrada.flatten(), 'Erro Relativo:': erroRelativoCentrada.flatten()}

    dataFrameAtrasada = pd.DataFrame(dataAtrasada)
    dataFrameAvancada = pd.DataFrame(dataAvancada)
    dataFrameCentrada = pd.DataFrame(dataCentrada)

    print("Atrasada: ")
    print(dataFrameAtrasada)
    print("Avancada: ")
    print(dataFrameAvancada)
    print("Centrada: ")
    print(dataFrameCentrada)

    plt.plot(x, difAtrasada, 'r-o', label='Dif. Atrasada')
    plt.plot(x, difAvancada, 'b-*', label='Dif. Avançada')
    plt.plot(x, difCentrada, 'k--^', label='Dif. Centrada')
    plt.plot(x, solucaoAnalitica, 'm-*', label='Solução Analítica')
    plt.legend()
    plt.title('Exercício E - Gráficos dos Métodos de Diferenças')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


exercicioA()
exercicioB()
exercicioC()
exercicioD()
exercicioE()