import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def metodo_de_taylor_ordem_1(f, t0, y0, h, num_steps):
    t = t0
    y = y0
    results = {'X': [t0], 'Metodo de Taylor Ordem 1': [y0], 'Exata': [exact_solution(t0)], 'Erro Local': [0]}
    
    for _ in range(num_steps):
        # Usando a expansão de Taylor de primeira ordem
        y_new = y + h * f(t, y)
        
        t = t + h
        
        exact_y = exact_solution(t)  # Valor exato da solução
        erro_local = (y - exact_y)  # Cálculo do erro local

        results['X'].append(t)
        results['Metodo de Taylor Ordem 1'].append(y_new)
        results['Exata'].append(exact_y)
        results['Erro Local'].append(erro_local)
        
        y = y_new
    
    return pd.DataFrame(results)

def metodo_de_taylor_ordem_2(f, t0, y0, h, num_steps):
    t = t0
    y = y0
    results = {'X': [t0], 'Metodo de Taylor Ordem 2': [y0], 'Exata': [exact_solution(t0)], 'Erro Local': [0]}
    
    for _ in range(num_steps):
        # Usando a expansão de Taylor de segunda ordem
        y_new = y + h * f(t, y) + (h**2 / 2) * (f(t, y) + h * f(t, y))  # Expandindo até a segunda ordem
        
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
    return np.exp(5*t) + 0.2

# Função que representa a derivada dy/dx = x - y + 2
def f(t, y):
    return 5*y - 1

# Condições iniciais
t0 = 0
y0 = 1.2
h = 0.1
num_steps = 20

# Chamando a função para resolver a EDO e criar o DataFrame com os valores
df_taylor1 = metodo_de_taylor_ordem_1(f, t0, y0, h, num_steps)
df_taylor2 = metodo_de_taylor_ordem_2(f, t0, y0, h, num_steps)

# Exibindo o DataFrame com os valores e o erro local
print(df_taylor1)
print(df_taylor2)

# Plotando os resultados
plt.figure(figsize=(10, 6))
plt.plot(df_taylor1['X'], df_taylor1['Metodo de Taylor Ordem 1'], label='Método de Taylor 1', marker='o')
plt.plot(df_taylor2['X'], df_taylor2['Metodo de Taylor Ordem 2'], label='Método de Taylor 2', marker='o')
plt.plot(df_taylor1['X'], df_taylor1['Exata'], label='Solução Exata', linestyle='--')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title("Solução da EDO usando o Método de Taylor e Euler's")
plt.grid(True)
plt.show()