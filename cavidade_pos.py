import numpy as np
from matplotlib import pyplot as plt
from numba import njit

''''''''''''''''''''''''''''''''''''''''''
''' ABRIR OS DADOS DO PROCESSAMENTO '''

caminho = "./resultados/"
id = 'Re1000'

numero_reynolds = np.load(f'{caminho}numero_reynolds.npz')
reynolds = numero_reynolds['reynolds']

malha = np.load(f'{caminho}malha.npz')
x = malha['x']
y = malha['y']
t = malha['t']

frames = np.load(f'{caminho}frames.npz')
frames_u = frames['frames_u']
frames_v = frames['frames_v']
frames_pressao = frames['frames_pressao']
frames_vorticidade = frames['frames_vorticidade']

N_x = len(x) - 1
N_y = len(y) - 1
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = t[1] - t[0]

''''''''''''''''''''''''''''''''''''''''''
''' VARIAVEIS DO REGIME PERMANENTE '''

tempo = t[-1]
u = np.transpose(frames_u[-1])
v = np.transpose(frames_v[-1])
p = np.transpose(frames_pressao[-1])
vorticidade = np.transpose(frames_vorticidade[-1])

modulo_velocidade = (u**2 + v**2)**0.5

''' Extrair a velocidade u em x=comprimento_x/2
N_x tem que ser par '''
# velocidade_meio = frames_u[-1][int(N_x/2), :]
# np.savez(f'{caminho}{N_x}velocidade_meio.npz', velocidade_meio=velocidade_meio)


@njit
def calcular_velocidade_unitaria(u, v, modulo_velocidade, u_unitario, v_unitario):
    for i in range(0, len(x) - 1):
        for j in range(0, len(y) - 1):
            if modulo_velocidade[i, j] > 0.0:
                u_unitario[i, j] = u[i, j] / modulo_velocidade[i, j]
                v_unitario[i, j] = v[i, j] / modulo_velocidade[i, j]
    return u_unitario, v_unitario


u_unitario = np.zeros((N_x + 1, N_y + 1))
v_unitario = np.zeros((N_x + 1, N_y + 1))
u_unitario, v_unitario = calcular_velocidade_unitaria(u, v, modulo_velocidade, u_unitario, v_unitario)


@njit
def calcular_funcao_corrente(u, v, N_x, N_y, dx, dy, dt, tol, psi):
    lamda = -(2/dx**2+2/dy**2)
    erro = 100
    while erro > tol:
        R_max = 0
        for i in range(1, N_x):
            for j in range(1, N_y):
                R = -(v[i, j] - v[i-1, j])/dx + (u[i, j]-u[i, j-1])/dy - \
                    ((psi[i+1, j] - 2*psi[i, j] + psi[i-1, j])/dx**2 + (psi[i, j+1] - 2*psi[i, j] + psi[i, j-1])/dy**2)
                R = R/lamda
                psi[i, j] = psi[i, j] + R
                if abs(R) > R_max:
                    R_max = abs(R)
        erro = R_max
    return psi


psi = np.zeros_like(frames_u[-1])
psi = calcular_funcao_corrente(frames_u[-1], frames_v[-1], N_x, N_y, dx, dy, dt, 1e-8, psi)

''''''''''''''''''''''''''''''''''''''''''
''' GRAFICOS DO REGIME PERMANENTE '''

plt.style.use(['science', 'notebook'])
xx, yy = np.meshgrid(x, y)

''' LINHAS DE CORRENTE 
Nas regioes de recirculacao, os valores e psi sao muito pequenos. Assim, é necessario prescrever manualmente valores
de psi para gerar um bom grafico das linhas de corrente. Seguem parametros para alguns Reynolds

Re = 1: niveis = [-0.078, -0.065, -0.05, -0.035, -0.02, -0.01, -0.002, -0.0003, -0.0001, 0, .00002, 0.00003]
Re = 10: niveis = [-0.078, -0.065, -0.05, -0.035, -0.02, -0.01, -0.002, -0.0003, -0.0001, 0, .00002, 0.00003]
Re = 100: niveis = [-0.078, -0.065, -0.05, -0.035, -0.02, -0.01, -0.002, -0.0003, -0.0001, 0, .00002, 0.00003]
Re = 1000: niveis = [-0.08, -0.075, -0.065, -0.05, -0.035, -0.02, -0.01, -0.002, -0.0003, 0, .00005,
          .0001, .00015, .0003, .0006, .0008]
Re = 5000:  niveis = [-0.067, -0.06, -0.05, -0.035, -0.02, -0.01, -0.0025, 0, 0.00005,
          0.00025, 0.0005, 0.0007, 0.001, 0.0015, 0.0016, 0.00175]
Re = 1000, comprimento_x=2: niveis = [-0.106, -0.1, -0.09, -0.08, -0.06, -0.035, -0.02, -0.01, -0.005, -0.002, -0.0003, 0, .00005,
          .0003, .0005, .001, 0.0015]
Re = 1000, comprimento_y=2: niveis = [-0.083, -0.075, -0.065, -0.05, -0.035, -0.02, -0.01, -0.002, -0.0003, 0, .00005,
          .0003, .001, 0.002, 0.003, 0.004]
'''

print(f'psi_minimo={np.min(psi)}, psi_maximo={np.max(psi)})')

grafico_corrente = plt.subplots(figsize=(3, 3))
niveis = [-0.08, -0.075, -0.065, -0.05, -0.035, -0.02, -0.01, -0.002, -0.0003, 0, .00005,
          .0001, .00015, .0003, .0006, .0008]
contour = plt.contour(xx, yy, np.transpose(psi), levels=niveis, colors='black', linestyles='solid', linewidths=1)
# plt.clabel(contour, inline=False, colors='r')

plt.axis('scaled')
plt.title(f't={tempo}, Re={reynolds}', fontsize=9)
# plt.xlabel('$x$', fontsize=10)
# plt.ylabel('$y$', fontsize=10)
# plt.xticks(fontsize=10), plt.yticks(fontsize=10)
plt.tick_params(axis='y', labelleft=False)
plt.tick_params(axis='x', labelbottom=False)

plt.savefig(f'{caminho}{id}corrente.png', format='png', dpi=700, bbox_inches='tight')
plt.show()

''' VETORES VELOCIDADE 
Plota-se o vetor unitario (direcao) da velocidade
'''
grafico_velocidade = plt.subplots(figsize=(2.8, 2.8))

# plt.pcolormesh(xx, yy, modulo_velocidade, cmap=plt.cm.BuGn)
plt.pcolormesh(xx, yy, modulo_velocidade, cmap=plt.cm.rainbow)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)

step = 3
plt.quiver(x[::step], y[::step], u_unitario[::step, ::step], v_unitario[::step, ::step], units='xy')
# plt.quiver(x, y, u_unitario, v_unitario)
plt.axis('scaled')
plt.title(f'Velocidade, t={tempo}, Re={reynolds}', fontsize=8)
plt.tick_params(axis='y', labelleft=False)
plt.tick_params(axis='x', labelbottom=False)

plt.savefig(f'{caminho}{id}vetores_velocidade.png', format='png', dpi=700, bbox_inches='tight')
plt.show()

''' CONTORNOS DE PRESSAO '''
grafico_pressao = plt.subplots(figsize=(2.7, 2.7))
plt.pcolormesh(xx, yy, p, cmap=plt.cm.rainbow)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
plt.axis('scaled')
plt.title(f'Pressão, t={tempo}, Re={reynolds}', fontsize=7)
plt.tick_params(axis='y', labelleft=False)
plt.tick_params(axis='x', labelbottom=False)

plt.savefig(f'{caminho}{id}pressao.png', format='png', dpi=700, bbox_inches='tight')
plt.show()

'''VORTICIDADE '''
grafico_vorticidade = plt.subplots(figsize=(2.8, 2.8))
plt.pcolormesh(xx, yy, vorticidade, cmap=plt.cm.rainbow)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)
plt.axis('scaled')
plt.title(f'Vorticidade, t={tempo}, Re={reynolds}', fontsize=8)
plt.tick_params(axis='y', labelleft=False)
plt.tick_params(axis='x', labelbottom=False)

plt.savefig(f'{caminho}{id}vorticidade.png', format='png', dpi=700, bbox_inches='tight')
plt.show()

'''' MODULO DA VELOCIDADE '''
# fig5 = plt.subplots()
# plt.pcolormesh(xx, yy, modulo_velocidade, cmap=plt.cm.jet)
# plt.axis('scaled')
# plt.title(f'Módulo da velocidade, t={tempo}, Re={reynolds}', fontsize=12)
# plt.xlabel('$x$', fontsize=12)
# plt.ylabel('$y$', fontsize=12)
# plt.xticks(fontsize=12), plt.yticks(fontsize=12)
# plt.colorbar()
# plt.show()

''' STREAMPLOT DO PYTHON '''
# fig2 = plt.subplots()
# plt.streamplot(x, y, u, v)
# plt.axis('scaled')
# plt.title(f'Streamplot, t={tempo}, Re={reynolds}', fontsize=12)
# plt.xlabel('$x$', fontsize=12)
# plt.ylabel('$y$', fontsize=12)
# plt.xticks(fontsize=12), plt.yticks(fontsize=12)
# plt.show()

''' CONTORNO DE VORTICIDADE '''
# print(f'vorticidade_minimo={np.min(vorticidade)}, vorticidade_maximo={np.max(vorticidade)})')
#
# contorno_vorticidade = plt.subplots()
# levels = [-32, -20, -10, -5, -2, -1.6, -1.5, -1.4,  -1, -0.5, 0, 0.5, 0.8, 1, 2, 5, 10, 15]
# contour2 = plt.contour(xx, yy, vorticidade, levels=levels, colors='black', linestyles='solid', linewidths=1)
# plt.clabel(contour2, inline=False, colors='r')
#
# plt.axis('scaled')
# plt.title(f'Contornos de vorticidade, t={tempo}, Re={reynolds}', fontsize=12)
# plt.xlabel('$x$', fontsize=12)
# plt.ylabel('$y$', fontsize=12)
# plt.xticks(fontsize=12), plt.yticks(fontsize=12)
#
# plt.show()

# ver todos os mapas: plt.colormaps()

''' FORCA NA TAMPA SUPERIOR '''
f = 0
for i in range(0, N_x+1):
    f += (2*psi[i, N_y] - 5*psi[i, N_y-1] + 4*psi[i, N_y-2] - psi[i, N_y-3])/dx
print(f)
