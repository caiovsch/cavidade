import numpy as np
from numba import njit

''''''''''''''''''''''''''''''''''''
''' PARAMETROS DE ENTRADA '''

''' PARAMETROS FISICOS '''
comprimento_x = 1
comprimento_y = 1
reynolds = 1000
tempo_final = 60
def calcular_velocidade_topo(x):
    # return np.ones_like(x)          # velocidade uniforme
    return np.sin(np.pi*x)**2       # evita descontinuidade
    # return np.sin(np.pi*x/2)**2     # para comprimento_x=2

''' PARAMETROS NUMERICOS '''
dt = 0.005
N_x = 100  # numero de divisoes (numero de pontos - 1)
N_y = 100

''' PARAMETROS POS-PROCESSAMENTO '''
numero_frames_tempo = 200

''''''''''''''''''''''''''''''''''''
''' PRE-PROCESSAMENTO '''

''' PASSOS '''
dx = comprimento_x/N_x
dy = comprimento_y/N_y

''' CHECA SE O PARAMETROS GARANTEM CONVERGENCIA '''
if dt >= dx:
    msg = 'Para estabilidade, dt<dx'
    raise RuntimeError(msg)

if dx >= 1/reynolds**0.5 or dy >= 1/reynolds**0.5:
    msg = 'Para estabilidade, dx<1/(Re**0.5), dy<1/(Re**0.5)'
    raise RuntimeError(msg)

if dt >= 1/4*reynolds*dx**2 or dt >= 1/4*reynolds*dy**2:
    msg = 'Para estabilidade, dt<1/4*reynolds*dx**2, dt<1/4*reynolds*dy**2'
    raise RuntimeError(msg)

''' MALHA '''
x = np.arange(0, comprimento_x+dx, dx)
print(x[int(N_x/2)])
y = np.arange(0, comprimento_y+dy, dy)
t = np.arange(0, tempo_final+dt, dt)

''' CONDICAO INICIAL '''
u = np.zeros([N_x+1, N_y+2])
velocidade_topo = calcular_velocidade_topo(x)
u[:, N_y] = 2*velocidade_topo
v = np.zeros([N_x+2, N_y+1])
p = np.zeros([N_x+2, N_y+2])

''' INICIALIZANDO AS VARIAVEIS AUXILIARES '''
u_star = np.zeros_like(u)
v_star = np.zeros_like(v)

''''''''''''''''''''''''''''''''''''
''' PROCESSAMENTO '''


@njit
def calcular_u_star(u, v, N_x, N_y, dx, dy, dt, u_star, velocidade_topo):

    for i in range(1, N_x):
        for j in range(0, N_y):
            C1 = 1/4 * (v[i, j+1] + v[i-1, j+1] + v[i, j] + v[i-1, j])
            R = -dt * (u[i, j] * (u[i+1, j] - u[i-1, j])/(2*dx) + C1 * (u[i, j+1] - u[i, j-1])/(2*dy)) + \
                dt/reynolds * ((u[i+1, j] - 2*u[i, j] + u[i-1, j])/dx**2 + (u[i, j+1] - 2*u[i, j] + u[i, j-1])/dy**2)
            u_star[i, j] = u[i, j] + R

    ''' FRONTEIRAS '''
    for j in range(0, N_y):
        u_star[0, j] = 0
        u_star[N_x, j] = 0
    for i in range(0, N_x+1):
        u_star[i, -1] = -u_star[i, 0]
        u_star[i, N_y] = 2 * velocidade_topo[i] - u_star[i, N_y - 1]

    return u_star


@njit
def calcular_v_star(u, v, N_x, N_y, dx, dy, dt, v_star):

    for i in range(0, N_x):
        for j in range(1, N_y):
            C2 = 1/4 * (u[i+1, j] + u[i, j] + u[i+1, j-1] + u[i, j-1])
            R = -dt * (C2 * (v[i+1, j] - v[i-1, j])/(2*dx) + v[i, j] * (v[i, j+1] - v[i, j-1])/(2*dy)) + \
                dt/reynolds * ((v[i+1, j] - 2*v[i, j] + v[i-1, j])/dx**2 + (v[i, j+1] - 2*v[i, j] + v[i, j-1])/dy**2)
            v_star[i, j] = v[i, j] + R

    ''' FRONTEIRAS '''
    for j in range(0, N_y+1):
        v_star[-1, j] = -v_star[0, j]
        v_star[N_x, j] = -v_star[N_x-1, j]
    for i in range(0, N_x):
        v_star[i, 0] = 0
        v_star[i, N_y] = 0

    return v_star


@njit
def calcular_pressao(u_star, v_star, N_x, N_y, dx, dy, dt, tol, p):

    R = 0.0
    erro = 1000
    while erro > tol:
        R_max = 0
        for i in range(0, N_x):
            for j in range(0, N_y):

                if i == 0 and j == 0:
                    lamda = -(1/dx**2 + 1/dy**2)
                    R = (u_star[i+1, j] - u_star[i, j])/dt/dx + (v_star[i, j+1] - v_star[i, j])/dt/dy - \
                        ((p[i+1, j] - p[i, j])/dx**2 + (p[i, j+1] - p[i, j])/dy**2)

                elif i == 0 and j == N_y-1:
                    lamda = -(1/dx**2 + 1/dy**2)
                    R = (u_star[i+1, j] - u_star[i, j])/dt/dx + (v_star[i, j+1] - v_star[i, j])/dt/dy - \
                        ((p[i+1, j] - p[i, j])/dx**2 + (-p[i, j] + p[i, j-1])/dy**2)

                elif i == N_x-1 and j == 0:
                    lamda = -(1/dx**2 + 1/dy**2)
                    R = (u_star[i+1, j] - u_star[i, j])/dt/dx + (v_star[i, j+1] - v_star[i, j])/dt/dy - \
                        ((-p[i, j] + p[i-1, j])/dx**2 + (p[i, j+1]-p[i, j])/dy**2)

                elif i == N_x-1 and j == N_y-1:
                    lamda = -(1/dx**2 + 1/dy**2)
                    R = (u_star[i+1, j] - u_star[i, j])/dt/dx + (v_star[i, j+1] - v_star[i, j])/dt/dy - \
                        ((-p[i, j] + p[i-1, j])/dx**2 + (-p[i, j] + p[i, j-1])/dy**2)

                elif i == 0 and j != 0 and j != N_y-1:
                    lamda = -(1/dx**2 + 2/dy**2)
                    R = (u_star[i+1, j] - u_star[i, j])/dt/dx + (v_star[i, j+1] - v_star[i, j])/dt/dy - \
                        ((p[i+1, j] - p[i, j])/dx**2 + (p[i, j+1] - 2*p[i, j] + p[i, j-1])/dy**2)

                elif i == N_x-1 and j != 0 and j != N_y-1:
                    lamda = -(1/dx**2 + 2/dy**2)
                    R = (u_star[i+1, j] - u_star[i, j])/dt/dx + (v_star[i, j+1] - v_star[i, j])/dt/dy - \
                        ((-p[i, j] + p[i-1, j])/dx**2 + (p[i, j+1] - 2*p[i, j] + p[i, j-1])/dy**2)

                elif j == 0 and i != 0 and i != N_x-1:
                    lamda = -(2/dx**2 + 1/dy**2)
                    R = (u_star[i+1, j] - u_star[i, j])/dt/dx + (v_star[i, j+1] - v_star[i, j])/dt/dy - \
                        ((p[i+1, j] - 2*p[i, j] + p[i-1, j])/dx**2 + (p[i, j+1] - p[i, j])/dy**2)

                elif j == N_y-1 and i != 0 and i != N_x-1:
                    lamda = -(2/dx**2 + 1/dy**2)
                    R = (u_star[i+1, j] - u_star[i, j])/dt/dx + (v_star[i, j+1] - v_star[i, j])/dt/dy - \
                        ((p[i+1, j] - 2*p[i, j] + p[i-1, j])/dx**2 + (-p[i, j] + p[i, j-1])/dy**2)

                else:
                    lamda = -(2/dx**2 + 2/dy**2)
                    R = (u_star[i+1, j] - u_star[i, j])/dt/dx + (v_star[i, j+1] - v_star[i, j])/dt/dy - \
                        ((p[i+1, j] - 2*p[i, j] + p[i-1, j])/dx**2 + (p[i, j+1] - 2*p[i, j] + p[i, j-1])/dy**2)

                R = R/lamda
                p[i, j] = p[i, j] + R

                if np.abs(R) > R_max:
                    R_max = np.abs(R)

        erro = R_max

    ''' FRONTEIRAS '''
    for i in range(0, N_x):
        p[i, -1] = p[i, 0]
        p[i, N_y] = p[i, N_y-1]
    for j in range(0, N_y):
        p[-1, j] = p[0, j]
        p[N_x, j] = p[N_x-1, j]
    p[-1, -1] = p[0, 0]
    p[-1, N_y] = p[0, N_y-1]
    p[N_x, -1] = p[N_x-1, 0]
    p[N_x, N_y] = p[N_x-1, N_y-1]

    return p


@njit
def calcular_u(u_star, p, N_x, N_y, dx, dt, u):
    for i in range(1, N_x):
        for j in range(-1, N_y+1):
            u[i, j] = u_star[i, j] - dt * (p[i, j] - p[i-1, j])/dx
    return u


@njit
def calcular_v(v_star, p, N_x, N_y, dy, dt, v):
    for i in range(-1, N_x+1):
        for j in range(1, N_y):
            v[i, j] = v_star[i, j] - dt * (p[i, j] - p[i, j-1])/dy
    return v


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


@njit
def interpolar_malha(u, v, p, N_x, N_y, uplot, vplot, pplot):
    for i in range(0, N_x + 1):
        for j in range(0, N_y + 1):
            uplot[i, j] = 1 / 2 * (u[i, j] + u[i, j - 1])
            vplot[i, j] = 1 / 2 * (v[i, j] + v[i - 1, j])
            pplot[i, j] = 1 / 4 * (p[i, j] + p[i - 1, j] + p[i, j - 1] + p[i - 1, j - 1])
    return uplot, vplot, pplot


@njit
def calcular_vorticidade(u, v, N_x, N_y, dx, dy, velocidade_topo, vorticidade):
    # diferenca avancada
    for i in range(0, N_x):
        for j in range(0, N_y):
            vorticidade[i, j] = (v[i + 1, j] - v[i, j]) / dx - (u[i, j + 1] - u[i, j]) / dy

    # topo
    for i in range(0, N_x):
        vorticidade[i, N_y] = (v[i+1, N_y] - v[i, N_y])/dx - (2*velocidade_topo[i] - 2*u[i, N_y])/dy
    # quina
    vorticidade[N_x, N_y] = -2*v[N_x, N_y]/dx - (2*velocidade_topo[N_x] - 2*u[N_x, N_y])/dy
    # direita
    for j in range(0, N_y):
        vorticidade[N_x, j] = -2 * v[N_x, j] / dx - (u[N_x, j + 1] - u[N_x, j]) / dy

    # diferenca centrada
    # for i in range(1, N_x):
    #     for j in range(1, N_y):
    #         vorticidade[i, j] = (v[i+1, j] - v[i-1, j])/2/dx - (u[i, j+1] - u[i, j-1])/2/dy

    return vorticidade


''' VARIAVEIS PARA SALVAR ALGUNS INSTANTES DE TEMPO '''
numero_passos_tempo = len(t) - 1
intervalo_captura = numero_passos_tempo // numero_frames_tempo
indices_tempo_selecionados = np.arange(0, len(t), intervalo_captura)

frames_pressao = np.zeros((numero_frames_tempo+1, N_x+1, N_y+1))
frames_u = np.zeros_like(frames_pressao)
frames_v = np.zeros_like(frames_pressao)
frames_vorticidade = np.zeros_like(frames_pressao)

''' AVANCO NO TEMPO '''

uplot = np.zeros((N_x + 1, N_y + 1))
vplot = np.zeros_like(uplot)
pplot = np.zeros_like(uplot)

frame_atual = 0
uplot, vplot, pplot = interpolar_malha(u, v, p, N_x, N_y, uplot, vplot, pplot)

indices_tempo_selecionados[0] = 0
frames_pressao[0] = pplot
frames_u[0] = uplot
frames_v[0] = vplot

vorticidade = np.zeros((N_x + 1, N_y + 1))

for k in range(1, len(t)):

    ''' CALCULO PARA CADA INSTANTE DE TEMPO '''
    u_star = calcular_u_star(u, v, N_x, N_y, dx, dy, dt, u_star, velocidade_topo)
    v_star = calcular_v_star(u, v, N_x, N_y, dx, dy, dt, v_star)
    p = calcular_pressao(u_star, v_star, N_x, N_y, dx, dy, dt, 1e-5, p)
    u = calcular_u(u_star, p, N_x, N_y, dx, dt, u)
    v = calcular_v(v_star, p, N_x, N_y, dx, dt, v)

    ''' SALVAR ALGUNS INSTANTES DE TEMPO '''
    if k % intervalo_captura == 0:

        frame_atual += 1

        uplot, vplot, pplot = interpolar_malha(u, v, p, N_x, N_y, uplot, vplot, pplot)
        frames_u[frame_atual] = uplot
        frames_v[frame_atual] = vplot
        frames_pressao[frame_atual] = pplot

        vorticidade = calcular_vorticidade(uplot, vplot, N_x, N_y, dx, dy, velocidade_topo, vorticidade)
        frames_vorticidade[frame_atual] = vorticidade

        print(f'tempo={indices_tempo_selecionados[frame_atual]*dt}')


''''''''''''''''''''''''''''''''''''
''' SALVAR DADOS PARA O POS-PROCESAMENTO '''
caminho = "./resultados/"

np.savez(f'{caminho}numero_reynolds.npz', reynolds=reynolds)
np.savez(f'{caminho}malha.npz', x=x, y=y, t=t)
np.savez(f'{caminho}frames.npz',
         indices_tempo_selecionados=indices_tempo_selecionados,
         frames_u=frames_u, frames_v=frames_v, frames_pressao=frames_pressao,
         frames_vorticidade=frames_vorticidade)
