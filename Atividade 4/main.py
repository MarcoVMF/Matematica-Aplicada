import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def exercicioB():
    a = 0
    b = 2
    h = 0.1
    n = int((b-a)/h)

    x = np.arange(a, b + h, h)

    eulerExplicito = np.zeros(n + 1)
    eulerImplicito = np.zeros(n + 1)
    difCentral = np.zeros(n + 1)
    regraTrapezios = np.zeros(n + 1)

    regraTrapezios[0] = 0.5
    eulerExplicito[0] = 0.5
    eulerImplicito[0] = 0.5
    difCentral[0] = 0.5
    difCentral[1] = 0.657414

    for i in range(n):
        eulerImplicito[i + 1] = eulerImplicito[i] * (1 + h) + h * (1 - (x[i] ** 2))
        eulerExplicito[i + 1] = (eulerExplicito[i] - h * (x[i + 1] ** 2) + h) / (1 - h)
        regraTrapezios[i + 1] = (regraTrapezios[i] + (h/2)*(regraTrapezios[i] - (x[i]**2) - (x[i+1]**2) + 2))/(1 - (h/2))

    for i in range(1, n):
        difCentral[i + 1] = difCentral[i - 1] + 2*h*(difCentral[i] - (x[i])**2 + 1)

    exata = ((x + 1)**2) - 0.5*np.exp(x)

    ea1 = np.abs(exata.T - eulerExplicito)
    ea2 = np.abs(exata.T - eulerImplicito)
    ea3 = np.abs(exata.T - difCentral.T)
    ea4 = np.abs(exata.T - regraTrapezios.T)
    er1 = np.abs((exata.T - eulerExplicito) / exata.T)
    er2 = np.abs((exata.T - eulerImplicito) / exata.T)
    er3 = np.abs((exata.T - difCentral.T) / exata.T)
    er4 = np.abs((exata.T - regraTrapezios.T) / exata.T)


    data1 = {"X:": x.T, "Euler Explicito: ": eulerExplicito, "Exata: ": exata.T, "EA1: ": ea1, "ER1: ": er1}
    data2 = {"X:": x.T, "Euler Implicito: ": eulerImplicito, "Exata: ": exata.T, "EA2: ": ea2, "ER2: ": er2}
    data3 = {"X:": x.T, "Diferenca Central: ": difCentral.T, "Exata: ": exata.T, "EA3: ": ea3, "ER3: ": er3}
    data4 = {"X:": x.T, "Regra dos Trapézios: ": regraTrapezios.T, "Exata: ": exata.T, "EA4: ": ea4, "ER4: ": er4}

    dataFrame1 = pd.DataFrame(data1)
    dataFrame2 = pd.DataFrame(data2)
    dataFrame3 = pd.DataFrame(data3)
    dataFrame4 = pd.DataFrame(data4)

    print("\nAvançado: ")
    print(dataFrame1)
    print("\nAtrasado: ")
    print(dataFrame2)
    print("\nCentrado: ")
    print(dataFrame3)
    print("\nTrapezios: ")
    print(dataFrame4)

    with pd.ExcelWriter('exercicioB.xlsx') as writer:
        dataFrame1.to_excel(writer, sheet_name='Diferenca Atrasada', index=False)
        dataFrame2.to_excel(writer, sheet_name='Diferenca Avancada', index=False)
        dataFrame3.to_excel(writer, sheet_name='Diferenca Centrada', index=False)
        dataFrame4.to_excel(writer, sheet_name='Regra dos Trapezios', index=False)



    plt.plot(x, exata, 'r-o', label='Exato: ')
    plt.plot(x, eulerExplicito, 'b--*', label='Euler Explicito: ')
    plt.plot(x, eulerImplicito, 'k--^', label='Euler Implicito: ')
    plt.plot(x, regraTrapezios, 'g-x', label='Regra dos Trapézios: ')
    plt.plot(x, difCentral, 'y--', label='Diferença Central: ')
    plt.legend()
    plt.title("Comparação dos Métodos: ")
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def exercicioC():
    a = 0
    b = 1
    h = 0.1
    n = int((b - a) / h)

    x = np.arange(a, b + h, h)

    eulerExplicito = np.zeros(n + 1)
    eulerImplicito = np.zeros(n + 1)
    difCentral = np.zeros(n + 1)
    regraTrapezios = np.zeros(n + 1)

    regraTrapezios[0] = 1
    eulerExplicito[0] = 1
    eulerImplicito[0] = 1
    difCentral[0] = 1
    difCentral[1] = 0.9125

    for i in range(n):
        eulerImplicito[i + 1] = (eulerImplicito[i] + h*x[i+1] + h)/(1 + 2*h)
        eulerExplicito[i + 1] = eulerExplicito[i]*(1-2*h) + h*x[i] + h
        regraTrapezios[i + 1] = (regraTrapezios[i] + (h/2)*(x[i] - 2*regraTrapezios[i] + 1 + x[i+1] + 1))/(1+h)

    for i in range(1, n):
        difCentral[i + 1] = difCentral[i - 1] + 2 * h * (x[i] - 2*difCentral[i] + 1)

    exata = (3*np.exp(-2*x) + 2*x + 1)/4

    ea1 = np.abs(exata.T - eulerExplicito)
    ea2 = np.abs(exata.T - eulerImplicito)
    ea3 = np.abs(exata.T - difCentral.T)
    ea4 = np.abs(exata.T - regraTrapezios.T)
    er1 = np.abs((exata.T - eulerExplicito) / exata.T)
    er2 = np.abs((exata.T - eulerImplicito) / exata.T)
    er3 = np.abs((exata.T - difCentral.T) / exata.T)
    er4 = np.abs((exata.T - regraTrapezios.T) / exata.T)

    data1 = {"X:": x.T, "Euler Explicito: ": eulerExplicito, "Exata: ": exata.T, "EA1: ": ea1, "ER1: ": er1}
    data2 = {"X:": x.T, "Euler Implicito: ": eulerImplicito, "Exata: ": exata.T, "EA2: ": ea2, "ER2: ": er2}
    data3 = {"X:": x.T, "Diferenca Central: ": difCentral.T, "Exata: ": exata.T, "EA3: ": ea3, "ER3: ": er3}
    data4 = {"X:": x.T, "Regra dos Trapézios: ": regraTrapezios.T, "Exata: ": exata.T, "EA4: ": ea4, "ER4: ": er4}

    dataFrame1 = pd.DataFrame(data1)
    dataFrame2 = pd.DataFrame(data2)
    dataFrame3 = pd.DataFrame(data3)
    dataFrame4 = pd.DataFrame(data4)

    print("\nAvançado: ")
    print(dataFrame1)
    print("\nAtrasado: ")
    print(dataFrame2)
    print("\nCentrado: ")
    print(dataFrame3)
    print("\nTrapezios: ")
    print(dataFrame4)

    with pd.ExcelWriter('exercicioC.xlsx') as writer:
        dataFrame1.to_excel(writer, sheet_name='Diferenca Atrasada', index=False)
        dataFrame2.to_excel(writer, sheet_name='Diferenca Avancada', index=False)
        dataFrame3.to_excel(writer, sheet_name='Diferenca Centrada', index=False)
        dataFrame4.to_excel(writer, sheet_name='Regra dos Trapezios', index=False)

    plt.plot(x, exata, 'r-o', label='Exato: ')
    plt.plot(x, eulerExplicito, 'b--*', label='Euler Explicito: ')
    plt.plot(x, eulerImplicito, 'k--^', label='Euler Implicito: ')
    plt.plot(x, regraTrapezios, 'g-x', label='Regra dos Trapézios: ')
    plt.plot(x, difCentral, 'y--', label='Diferença Central: ')
    plt.legend()
    plt.title("Comparação dos Métodos: ")
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

exercicioB()
exercicioC()