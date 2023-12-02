import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def derivada(t, y):
    return ((2 * (t ** 3)) + 2) * y


def exata(t):
    return np.exp((t ** 4) / 2) * np.exp(2 * t)


def euler_explicito(f, t0, y0, h, n):
    t = t0
    y = y0

    resultados = {'x': [t0], 'Euler Explicito': [y0], 'Exata': [exata(t0)], 'Erro': [0]}

    for _ in range(n):
        y_novo = y + h * f(t, y)
        t = t + h

        valor_exato = exata(t)
        erro = abs(valor_exato - y_novo)

        resultados['x'].append(t)
        resultados['Euler Explicito'].append(y_novo)
        resultados['Exata'].append(valor_exato)
        resultados['Erro'].append(erro)

        y = y_novo

    return pd.DataFrame(resultados)


def euler_implicito(f, t0, y0, h, n):
    t = t0
    y = y0
    resultado = {'x': [t0], 'Euler Implicito': [y0], 'Exata': [exata(t0)], 'Erro': [0]}

    for _ in range(n):
        y_novo = y + h * f(t + h, y + h * f(t, y))
        t = t + h

        valor_exato = exata(t)
        erro = abs(valor_exato - y_novo)

        resultado['x'].append(t)
        resultado['Euler Implicito'].append(y_novo)
        resultado['Exata'].append(valor_exato)
        resultado['Erro'].append(erro)

        y = y_novo

    return pd.DataFrame(resultado)


def regra_trapezio(f, t0, y0, h, n):
    t = t0
    y = y0
    resultado = {'x': [t0], 'Regra do Trapezio': [y0], 'Exata': [exata(t0)], 'Erro': [0]}

    for _ in range(n):
        y_novo = y + (h / 2) * (f(t, y) + f(t + h, y + h * f(t, y)))
        t = t + h

        valor_exato = exata(t)
        erro = abs(valor_exato - y_novo)

        resultado['x'].append(t)
        resultado['Regra do Trapezio'].append(y_novo)
        resultado['Exata'].append(valor_exato)
        resultado['Erro'].append(erro)

        y = y_novo

    return pd.DataFrame(resultado)


def taylor_ordem_2(f, t0, y0, h, n):
    t = t0
    y = y0
    resultado = {'x': [t0], 'Taylor Ordem 2': [y0], 'Exata': [exata(t0)], 'Erro': [0]}

    for _ in range(n):
        y_novo = y + h * f(t, y) + (h ** 2 / 2) * (f(t, y) + h * f(t, y))

        t = t + h

        valor_exato = exata(t)
        erro = abs(valor_exato - y_novo)

        resultado['x'].append(t)
        resultado['Taylor Ordem 2'].append(y_novo)
        resultado['Exata'].append(valor_exato)
        resultado['Erro'].append(erro)

        y = y_novo

    return pd.DataFrame(resultado)


def euler_modificado(f, t0, y0, h, n):
    t = t0
    y = y0
    resultado = {'x': [t0], 'Euler Modificado': [y0], 'Exata': [exata(t0)], 'Erro': [0]}

    for _ in range(n):
        y_pred = y + h * f(t, y)
        t = t + h
        y = y + 0.5 * h * (f(t, y) + f(t, y_pred))

        valor_exato = exata(t)
        erro = abs(valor_exato - y)

        resultado['x'].append(t)
        resultado['Euler Modificado'].append(y)
        resultado['Exata'].append(valor_exato)
        resultado['Erro'].append(erro)

    return pd.DataFrame(resultado)


def euler_aperfeicoado(f, t0, y0, h, n):
    t = t0
    y = y0
    resultado = {'x': [t0], 'Euler Aprimorado': [y0], 'Exata': [exata(t0)], 'Erro': [0]}

    for _ in range(n):
        k1 = h * f(t, y)
        k2 = h * f(t + h, y + k1)

        y = y + 0.5 * (k1 + k2)
        t = t + h

        valor_exato = exata(t)
        erro = abs(valor_exato - y)

        resultado['x'].append(t)
        resultado['Euler Aprimorado'].append(y)
        resultado['Exata'].append(valor_exato)
        resultado['Erro'].append(erro)

    return pd.DataFrame(resultado)


def runge_kutta_order_3(f, t0, y0, h, n):
    t = t0
    y = y0
    resultado = {'x': [t0], 'Runge Kutta 3': [y0], 'Exata': [exata(t0)], 'Erro': [0]}

    for _ in range(n):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + h, y - k1 + 2 * k2)

        y = y + (k1 + 4 * k2 + k3) / 6.0
        t = t + h

        valor_exato = exata(t)
        erro = abs(valor_exato - y)

        resultado['x'].append(t)
        resultado['Runge Kutta 3'].append(y)
        resultado['Exata'].append(valor_exato)
        resultado['Erro'].append(erro)

    return pd.DataFrame(resultado)


def runge_kutta_order_4(f, t0, y0, h, n):
    t = t0
    y = y0
    resultado = {'x': [t0], 'Runge Kutta 4': [y0], 'Exata': [exata(t0)], 'Erro': [0]}

    for _ in range(n):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)

        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t = t + h

        valor_exato = exata(t)
        erro = abs(valor_exato - y)

        resultado['x'].append(t)
        resultado['Runge Kutta 4'].append(y)
        resultado['Exata'].append(valor_exato)
        resultado['Erro'].append(erro)

    return pd.DataFrame(resultado)


def previsor_euler(f, t, y, h):
    return y + h * f(t, y)


def corretor_trapezio(f, t, y, y_pred, h):
    return y + (h / 2) * (f(t, y) + f(t + h, y_pred))


def previsor_corretor(f, t0, y0, h, n, tolerancia=0.01):
    t = [t0]
    y = [y0]
    resultado = {'x': [t0], 'Previsor-Corretor': [y0], 'Exata': [exata(t0)], 'Erro': [0]}

    for i in range(n):
        t_atual = t[i]
        y_atual = y[i]
        t_proximo = t_atual + h

        y_preditor = previsor_euler(f, t_atual, y_atual, h)

        while True:
            y_corretor = corretor_trapezio(f, t_atual, y_atual, y_preditor, h)

            if abs(y_corretor - y_preditor) < tolerancia:
                break
            else:
                y_preditor = y_corretor

        valor_exato = exata(t_proximo)
        erro = abs(valor_exato - y_corretor)

        y.append(y_corretor)
        t.append(t_proximo)

        resultado['x'].append(t_proximo)
        resultado['Previsor-Corretor'].append(y_corretor)
        resultado['Exata'].append(valor_exato)
        resultado['Erro'].append(erro)

    return pd.DataFrame(resultado)


def execucao(h, malha):
    y0 = 1
    t0 = 0
    n = 10

    t = np.arange(t0, t0 + n * h + h, h)

    y_euler_explicito = euler_explicito(derivada, t0, y0, h, n)
    y_euler_implicito = euler_implicito(derivada, t0, y0, h, n)
    y_regra_trapezio = regra_trapezio(derivada, t0, y0, h, n)
    y_taylor_ordem_2 = taylor_ordem_2(derivada, t0, y0, h, n)
    y_euler_modificado = euler_modificado(derivada, t0, y0, h, n)
    y_euler_aperfeicoado = euler_aperfeicoado(derivada, t0, y0, h, n)
    y_runge_kutta_order_3 = runge_kutta_order_3(derivada, t0, y0, h, n)
    y_runge_kutta_order_4 = runge_kutta_order_4(derivada, t0, y0, h, n)
    y_previsor_corretor = previsor_corretor(derivada, t0, y0, h, n)

    erros = {"Euler Explicito": y_euler_explicito['Erro'], "Euler Implicito": y_euler_implicito['Erro'],
             "Regra do Trapezio": y_regra_trapezio['Erro'], "Taylor Ordem 2": y_taylor_ordem_2['Erro'],
             "Euler Modificado": y_euler_modificado['Erro'], "Euler Aprimorado": y_euler_aperfeicoado['Erro'],
             "Runge Kutta 3": y_runge_kutta_order_3['Erro'], "Runge Kutta 4": y_runge_kutta_order_4['Erro'],
             "Previsor-Corretor": y_previsor_corretor['Erro']}

    df_erros = pd.DataFrame(erros)

    df_erros.to_excel('Erros' + malha + '.xlsx', index=False)

    plt.title("Gráfico de Aproximações")
    plt.plot(t, y_euler_explicito['Euler Explicito'], label='Euler Explicito')
    plt.plot(t, y_euler_implicito['Euler Implicito'], label='Euler Implicito')
    plt.plot(t, y_regra_trapezio['Regra do Trapezio'], label='Regra do Trapezio')
    plt.plot(t, y_taylor_ordem_2['Taylor Ordem 2'], label='Taylor Ordem 2')
    plt.plot(t, y_euler_modificado['Euler Modificado'], label='Euler Modificado')
    plt.plot(t, y_euler_aperfeicoado['Euler Aprimorado'], label='Euler Aprimorado')
    plt.plot(t, y_runge_kutta_order_3['Runge Kutta 3'], label='Runge Kutta 3')
    plt.plot(t, y_runge_kutta_order_4['Runge Kutta 4'], label='Runge Kutta 4')
    plt.plot(y_previsor_corretor['x'], y_previsor_corretor['Previsor-Corretor'], label='Previsor-Corretor')
    plt.plot(t, y_euler_explicito['Exata'], label='Exata', linestyle='--')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.savefig('Grafico-Aprox' + malha + '.png')
    plt.show()

    plt.title("Gráfico de Erros")
    plt.plot(t, y_euler_explicito['Erro'], label='Euler Explicito')
    plt.plot(t, y_euler_implicito['Erro'], label='Euler Implicito')
    plt.plot(t, y_regra_trapezio['Erro'], label='Regra do Trapezio')
    plt.plot(t, y_taylor_ordem_2['Erro'], label='Taylor Ordem 2')
    plt.plot(t, y_euler_modificado['Erro'], label='Euler Modificado')
    plt.plot(t, y_euler_aperfeicoado['Erro'], label='Euler Aprimorado')
    plt.plot(t, y_runge_kutta_order_3['Erro'], label='Runge Kutta 3')
    plt.plot(t, y_runge_kutta_order_4['Erro'], label='Runge Kutta 4')
    plt.plot(y_previsor_corretor['x'], y_previsor_corretor['Erro'], label='Previsor-Corretor')
    plt.xlabel('t')
    plt.ylabel('Erro')
    plt.legend(loc='lower left')
    plt.grid()
    plt.savefig('Grafico-Erro' + malha + '.png')
    plt.show()


def main():
    execucao(0.1, "-Malha-0.1")
    execucao(0.05, "-Malha-0.05")


if __name__ == '__main__':
    main()