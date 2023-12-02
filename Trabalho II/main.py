import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Juros Compostos e Inflação

def derivada(r, C):
    return r*C


def exata(r, C, t):
    return C*np.exp(r*t)


def euler(r, C, t, h):
    y = np.zeros(len(t))
    y[0] = C
    for i in range(1, len(t)):
        y[i] = y[i-1] + h*derivada(r, y[i-1])

    erro = np.zeros(len(t))
    for i in range(1, len(t)):
        erro[i] = abs(y[i] - exata(r, C, t[i]))

    return y, erro


def euler_modificado(r, C, t, h):
    y = np.zeros(len(t))
    y[0] = C
    for i in range(1, len(t)):
        y[i] = y[i-1] + h*derivada(r, y[i-1])
        y[i] = y[i-1] + h*derivada(r, y[i])

    #Calculando o erro
    erro = np.zeros(len(t))
    for i in range(1, len(t)):
        erro[i] = abs(y[i] - exata(r, C, t[i]))

    return y, erro


def runge_kutta_ordem_4(r, C, t, h):
    y = np.zeros(len(t))
    y[0] = C
    for i in range(1, len(t)):
        k1 = h*derivada(r, y[i-1])
        k2 = h*derivada(r, y[i-1] + k1/2)
        k3 = h*derivada(r, y[i-1] + k2/2)
        k4 = h*derivada(r, y[i-1] + k3)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4)/6
    # Calculando o erro
    erro = np.zeros(len(t))
    for i in range(1, len(t)):
        erro[i] = abs(y[i] - exata(r, C, t[i]))

    return y, erro


def plot_graphs(t, y_exata, y_euler, y_euler_modificado, y_runge_kutta, grafico_nome ,zoom_y=None, zoom_x=None):
    plt.plot(t, y_exata, label='Exata', color='blue', linestyle='-')
    plt.plot(t, y_euler, label='Euler', color='green', linestyle='--')
    plt.plot(t, y_euler_modificado, label='Euler Modificado', color='red', linestyle='-.')
    plt.plot(t, y_runge_kutta, label='Runge Kutta 4', color='purple', linestyle=':')
    plt.grid()
    plt.xlabel(" Tempo ( Meses ) ")
    plt.ylabel(" Dinheiro ( R$1000 ) ")
    plt.legend()
    if zoom_x is not None and zoom_y is not None:
        plt.xlim(zoom_x[0], zoom_x[1])
        plt.ylim(zoom_y[0], zoom_y[1])
    else:
        plt.xlim(zoom_x)
        plt.ylim(zoom_y)
    plt.savefig(grafico_nome)
    plt.show()


def execucao(h):

    r = 0.0975                  # 9.75% ao mes
    C = 3                       # 3 mil reais
    t = np.arange(0, 12, h)     # 1 ano

    y_exata = exata(r, C, t)
    y_euler, erro_euler = euler(r, C, t, h)
    y_euler_modificado, erro_modificado = euler_modificado(r, C, t, h)
    y_runge_kutta, erro_kutta = runge_kutta_ordem_4(r, C, t, h)

    # DataFrame

    df_euler = pd.DataFrame({'Tempo': t, 'Exata': y_exata, 'Euler': y_euler, 'Erro Euler': erro_euler})
    df_euler_modificado = pd.DataFrame({'Tempo': t, 'Exata': y_exata, 'Euler Modificado': y_euler_modificado, 'Erro Euler Modificado': erro_modificado})
    df_runge_kutta = pd.DataFrame({'Tempo': t, 'Exata': y_exata, 'Runge Kutta': y_runge_kutta, 'Erro Runge Kutta': erro_kutta})


    df_euler.to_excel('euler' + "-" + "h=" + str(h) + '.xlsx')
    df_euler_modificado.to_excel('euler_modificado' + "-" + "h=" + str(h) + '.xlsx')
    df_runge_kutta.to_excel('runge_kutta' + "-" + "h=" + str(h) + '.xlsx')



    plot_graphs(t, y_exata, y_euler, y_euler_modificado, y_runge_kutta, grafico_nome='grafico' + "-" + "h=" + str(h) + '.png')
    plot_graphs(t, y_exata, y_euler, y_euler_modificado, y_runge_kutta, grafico_nome='graficoZoom' + "-" + "h=" + str(h) + '.png', zoom_x=[10, 11], zoom_y=[8, 9])


def main():
    execucao(1)
    execucao(0.1)


if __name__ == '__main__':
    main()