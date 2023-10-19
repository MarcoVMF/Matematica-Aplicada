import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def runge_kutta_order_3(f, t0, y0, h, num_steps):
    t = t0
    y = y0
    results = {'t': [t0], 'y': [y0]}
    
    for _ in range(num_steps):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + h, y - k1 + 2 * k2)
        
        y = y + (k1 + 4 * k2 + k3) / 6.0
        t = t + h
        
        results['t'].append(t)
        results['y'].append(y)
    
    return pd.DataFrame(results)

def runge_kutta_order_4(f, t0, y0, h, num_steps):
    t = t0
    y = y0
    results = {'t': [t0], 'y': [y0]}
    
    for _ in range(num_steps):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t = t + h
        
        results['t'].append(t)
        results['y'].append(y)
    
    return pd.DataFrame(results)

# Função que representa a derivada dy/dt = -2ty
def f(t, y):
    return -2 * t * y

# Solução exata
def exact_solution(t):
    return np.exp(-t**2)

# Condições iniciais
t0 = 0
y0 = 1.0

# Tamanho do passo e número de iterações
h = 0.1
num_steps = 100

# Chamando a função para resolver a EDO e criar o DataFrame
df = runge_kutta_order_3(f, t0, y0, h, num_steps)

# Criando valores de t para a solução exata
t_exact = np.linspace(t0, t0 + num_steps * h, 100)
y_exact = exact_solution(t_exact)

# Plotando o gráfico
plt.plot(df['t'], df['y'], label='Runge-Kutta 3rd Order')
plt.plot(t_exact, y_exact, label='Exact Solution', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solução da EDO usando Runge-Kutta de ordem 3 e Solução Exata')
plt.legend()
plt.show()
