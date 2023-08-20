import numpy as np
import matplotlib.pyplot as plt

f = lambda x: np.log(x)
df = lambda x: 1/x

inicio = 0
fim = 50
n = 20

h = (fim - inicio) / n

x = np.arange(inicio, fim + h, h)
x = x.reshape(-1, 1)

difAtrasada = (f(x + h) - f(x)) / h
difAvancada = (f(x) - f(x - h)) / h
difCentrada = (f(x + h) - f(x - h)) / (2 * h)
solucao_analitica = df(x)

erroAtrasada = np.abs((solucao_analitica - difAtrasada.flatten()) / solucao_analitica)
erroAvancada = np.abs((solucao_analitica - difAvancada.flatten()) / solucao_analitica)
erroCentrada = np.abs((solucao_analitica - difCentrada.flatten()) / solucao_analitica)

plt.plot(x, difAtrasada, 'r-o', label='Dif. Atrasada')
plt.plot(x, difAvancada, 'b-*', label='Dif. Avançada')
plt.plot(x, difCentrada, 'k--^', label='Dif. Centrada')
plt.plot(x, solucao_analitica, 'm-*', label='Solução Analítica')

plt.legend()
plt.title('Gráficos dos Métodos de Diferenças')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
