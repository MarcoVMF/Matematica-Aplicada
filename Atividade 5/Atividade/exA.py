import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def metodo_de_taylor_ordem_2(f, t0, y0, h, num_steps):
    t = t0
    y = y0
    results = {'X': [t0], 'Metodo de Taylor Ordem 2': [y0], 'Exata': [exact_solution(t0)], 'Erro Local': [0]}

    for _ in range(num_steps):
        # Usando a expansão de Taylor de segunda ordem
        y_new = y + h * f(t, y) + (h ** 2 / 2) * (f(t, y) + h * f(t, y))  # Expandindo até a segunda ordem

        t = t + h

        exact_y = exact_solution(t)  # Valor exato da solução
        erro_local = (y - exact_y)  # Cálculo do erro local

        results['X'].append(t)
        results['Metodo de Taylor Ordem 2'].append(y_new)
        results['Exata'].append(exact_y)
        results['Erro Local'].append(erro_local)

        y = y_new

    return pd.DataFrame(results)


# Solução exata
def exact_solution(t):
    return np.exp(-t) + t + 1


# Função que representa a derivada dy/dx = x - y + 2
def f(t, y):
    return t - y + 2


# Condições iniciais
t0 = 0
y0 = 2.0

# Tamanho do passo e número de iterações
num_subintervals = 5
t_max = 1.0  # Intervalo de 0 a 1
h = (t_max - t0) / num_subintervals
num_steps = num_subintervals

# Chamando a função para resolver a EDO e criar o DataFrame com os valores
df_taylor = metodo_de_taylor_ordem_2(f, t0, y0, h, num_steps)
h = 0.1
n = 10
df_euler_modificado = euler_modificado(f, t0, y0, h, n)
df_euler_aprimorado = euler_aprimorado(f, t0, y0, h, n)

# Exibindo o DataFrame com os valores e o erro local

with pd.ExcelWriter('exercicioA.xlsx') as writer:
    df_taylor.to_excel(writer, sheet_name='Taylor', index=False)
    df_euler_modificado.to_excel(writer, sheet_name='Runge Modificado', index=False)
    df_euler_aprimorado.to_excel(writer, sheet_name='Euler Aprimorado', index=False)


# Plotando os resultados
plt.figure(figsize=(10, 6))
plt.plot(df_taylor['X'], df_taylor['Metodo de Taylor Ordem 2'], label='Método de Taylor', marker='o')
plt.plot(df_euler_modificado['X'], df_euler_modificado['Euler Modificado'], label='Euler Modificado', marker='x')
plt.plot(df_euler_aprimorado['X'], df_euler_aprimorado['Euler Aprimorado'], label='Euler Aprimorado', marker='*')
plt.plot(df_taylor['X'], df_taylor['Exata'], label='Solução Exata', linestyle='--')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title("Solução da EDO usando o Método de Taylor e Euler's")
plt.grid(True)
plt.savefig('exercicioA.png')
plt.show()
