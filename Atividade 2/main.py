import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def exercicio1():
    print("EXERCÍCIO 1 =====================================================================================================================")
    cossec = lambda x: 1 / np.sin(x)
    cotg = lambda x: 1 / np.tan(x)

    f = lambda x: np.exp(x) * cossec(x)
    df = lambda x: (np.exp(x) * cossec(x))-(np.exp(x) * cossec(x) * cotg(x))
    inicio = 0.2
    fim = 1.5
    h = 0.1

    x = np.arange(inicio, fim + h, h)
    x = x.reshape(-1, 1)

    difAtrasada = (f(x + h) - f(x)) / h
    difAvancada = (f(x) - f(x - h)) / h
    difCentrada = (f(x + h) - f(x - h)) / (2 * h)
    solucaoAnalitica = df(x)

    erroRelativoAtrasada = np.abs((solucaoAnalitica - difAtrasada) / solucaoAnalitica)
    erroRelativoAvancada = np.abs((solucaoAnalitica - difAvancada) / solucaoAnalitica)
    erroRelativoCentrada = np.abs((solucaoAnalitica - difCentrada) / solucaoAnalitica)

    dataAtrasada = {'Atrasada:': difAtrasada.flatten(), 'Erro Relativo:': erroRelativoAtrasada.flatten()}
    dataAvancada = {'Avancada:': difAvancada.flatten(), 'Erro Relativo:': erroRelativoAvancada.flatten()}
    dataCentrada = {'Centrada:': difCentrada.flatten(), 'Erro Relativo:': erroRelativoCentrada.flatten()}

    dataFrameAtrasada = pd.DataFrame(dataAtrasada)
    dataFrameAvancada = pd.DataFrame(dataAvancada)
    dataFrameCentrada = pd.DataFrame(dataCentrada)

    print("Atrasada: ")
    print(dataFrameAtrasada)
    print("Avancada: ")
    print(dataFrameAvancada)
    print("Centrada: ")
    print(dataFrameCentrada)

    with pd.ExcelWriter('exercicio1.xlsx') as writer:
        dataFrameAvancada.to_excel(writer, sheet_name='Diferenca Atrasada', index=False)
        dataFrameAtrasada.to_excel(writer, sheet_name='Diferenca Avancada', index=False)
        dataFrameCentrada.to_excel(writer, sheet_name='Diferenca Centrada', index=False)


def exercicio2():
    print("EXERCÍCIO 2 =====================================================================================================================")
    sec = lambda x: 1 / np.cos(x)

    f = lambda x: np.log(x) * np.tan(x)
    df = lambda x: (np.tan(x) / x) + (np.log(x)*(sec(x)**2))
    inicio = 0.5
    fim = 1.2
    h = 0.1

    x = np.arange(inicio, fim + h, h)
    x = x.reshape(-1, 1)

    difAtrasada = (f(x + h) - f(x)) / h
    difAvancada = (f(x) - f(x - h)) / h
    difCentrada = (f(x + h) - f(x - h)) / (2 * h)
    solucaoAnalitica = df(x)

    erroRelativoAtrasada = np.abs((solucaoAnalitica - difAtrasada) / solucaoAnalitica)
    erroRelativoAvancada = np.abs((solucaoAnalitica - difAvancada) / solucaoAnalitica)
    erroRelativoCentrada = np.abs((solucaoAnalitica - difCentrada) / solucaoAnalitica)

    dataAtrasada = {'Atrasada:': difAtrasada.flatten(),'Erro Relativo:': erroRelativoAtrasada.flatten()}
    dataAvancada = {'Avancada:': difAvancada.flatten(),'Erro Relativo:': erroRelativoAvancada.flatten()}
    dataCentrada = {'Centrada:': difCentrada.flatten(),'Erro Relativo:': erroRelativoCentrada.flatten()}

    dataFrameAtrasada = pd.DataFrame(dataAtrasada)
    dataFrameAvancada = pd.DataFrame(dataAvancada)
    dataFrameCentrada = pd.DataFrame(dataCentrada)

    print("Atrasada: ")
    print(dataFrameAtrasada)
    print("Avancada: ")
    print(dataFrameAvancada)
    print("Centrada: ")
    print(dataFrameCentrada)

    with pd.ExcelWriter('exercicio2.xlsx') as writer:
        dataFrameAvancada.to_excel(writer, sheet_name='Diferenca Atrasada', index=False)
        dataFrameAtrasada.to_excel(writer, sheet_name='Diferenca Avancada', index=False)
        dataFrameCentrada.to_excel(writer, sheet_name='Diferenca Centrada', index=False)

exercicio1()
exercicio2()