import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def runge_kutta_order_3(f, t0, y0, h, num_steps):
    t = t0
    y = y0
    results = {'X': [t0], 'Runge Kutta 3': [y0], 'Exata': [exact_solution(t0)], 'Erro Local': [0]}

    for _ in range(num_steps):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + h, y - k1 + 2 * k2)

        y = y + (k1 + 4 * k2 + k3) / 6.0
        t = t + h

        exact_y = exact_solution(t)  # Valor exato da solução
        erro_local = (exact_y - y)  # Cálculo do erro local

        results['X'].append(t)
        results['Runge Kutta 3'].append(y)
        results['Exata'].append(exact_y)
        results['Erro Local'].append(erro_local)

    return pd.DataFrame(results)

def runge_kutta_order_4(f, t0, y0, h, num_steps):
    t = t0
    y = y0
    results = {'X': [t0], 'Runge Kutta 4': [y0], 'Exata': [exact_solution(t0)], 'Erro Local': [0]}

    for _ in range(num_steps):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)

        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t = t + h

        exact_y = exact_solution(t)  # Valor exato da solução
        erro_local = (exact_y - y)  # Cálculo do erro local

        results['X'].append(t)
        results['Runge Kutta 4'].append(y)
        results['Exata'].append(exact_y)
        results['Erro Local'].append(erro_local)

    return pd.DataFrame(results)

def euler_modificado(f, t0, y0, h, num_steps):
    t = t0
    y = y0
    results = {'X': [t0], 'Euler Modificado': [y0], 'Exata': [exact_solution(t0)], 'Erro Local': [0]}

    for _ in range(num_steps):
        y_pred = y + h * f(t, y)
        t = t + h
        y = y + 0.5 * h * (f(t, y) + f(t, y_pred))

        exact_y = exact_solution(t)  # Valor exato da solução
        erro_local = (exact_y - y)  # Cálculo do erro local

        results['X'].append(t)
        results['Euler Modificado'].append(y)
        results['Exata'].append(exact_y)
        results['Erro Local'].append(erro_local)

    return pd.DataFrame(results)


def euler_aprimorado(f, t0, y0, h, num_steps):
    t = t0
    y = y0
    results = {'X': [t0], 'Euler Aprimorado': [y0], 'Exata': [exact_solution(t0)], 'Erro Local': [0]}

    for _ in range(num_steps):
        k1 = h * f(t, y)
        k2 = h * f(t + h, y + k1)

        y = y + 0.5 * (k1 + k2)
        t = t + h

        exact_y = exact_solution(t)  # Valor exato da solução
        erro_local = (exact_y - y)  # Cálculo do erro local

        results['X'].append(t)
        results['Euler Aprimorado'].append(y)
        results['Exata'].append(exact_y)
        results['Erro Local'].append(erro_local)

    return pd.DataFrame(results)

# Função que representa a derivada dy/dt = -2ty
def f(t, y):
    return np.exp(-t) - 2*y


# Solução exata
def exact_solution(t):
    return np.exp(-t) + 2*np.exp(-2*t)


# Condições iniciais
t0 = 0
y0 = 3

# Tamanho do passo e número de iterações
h = 0.1
num_steps = 100

# Chamando a função para resolver a EDO e criar o DataFrame
dfKuttaOrder3 = runge_kutta_order_3(f, t0, y0, h, num_steps)
dfKuttaOrder4 = runge_kutta_order_4(f, t0, y0, h, num_steps)
dfEulerModificado = euler_modificado(f, t0, y0, h, num_steps)
dfEulerAprimorado = euler_aprimorado(f, t0, y0, h, num_steps)

# Criando valores de t para a solução exata
t_exact = np.linspace(t0, t0 + num_steps * h, 100)
y_exact = exact_solution(t_exact)

with pd.ExcelWriter('exercicioA.xlsx') as writer:
    dfKuttaOrder3.to_excel(writer, sheet_name='Runge Kutta 3', index=False)
    dfKuttaOrder4.to_excel(writer, sheet_name='Runge Kutta 4', index=False)
    dfEulerModificado.to_excel(writer, sheet_name='Euler Modificado', index=False)
    dfEulerAprimorado.to_excel(writer, sheet_name='Euler Aprimorado', index=False)


# Plotando o gráfico
plt.plot(dfKuttaOrder3['X'], dfKuttaOrder3['Runge Kutta 3'], label='Runge-Kutta 3 Ordem')
plt.plot(dfKuttaOrder4['X'], dfKuttaOrder4['Runge Kutta 4'], label='Runge-Kutta 4 Ordem')
plt.plot(dfEulerModificado['X'], dfEulerModificado['Euler Modificado'], label='Euler Modificado')
plt.plot(dfEulerAprimorado['X'], dfEulerAprimorado['Euler Aprimorado'], label='Euler Apromorado')
plt.plot(t_exact, y_exact, label='Exact Solution', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solução da EDO usando Runge-Kutta, Euler, Solução Exata')
plt.legend()
plt.savefig('exercicioA.png')
plt.show()