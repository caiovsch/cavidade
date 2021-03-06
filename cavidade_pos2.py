import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

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
indices_tempo_selecionados = frames['indices_tempo_selecionados']
frames_u = frames['frames_u']
frames_v = frames['frames_v']
frames_pressao = frames['frames_pressao']
frames_vorticidade = frames['frames_vorticidade']

frames_modulo_velocidade = (frames_u**2 + frames_v**2)**0.5

dt = t[1] - t[0]
xx, yy = np.meshgrid(x, y)


def mapa_filme(k):

    plt.clf()

    plt.pcolormesh(xx, yy, np.transpose(frames_para_animar[k]), cmap=plt.cm.jet)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    cb.ax.tick_params(labelsize=12)

    plt.title(f'{titulo}, Re={reynolds}, t={dt*indices_tempo_selecionados[k]:.3f}', fontsize=12)
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel('$y$', fontsize=12)
    plt.axis('scaled')
    plt.xticks(fontsize=12), plt.yticks(fontsize=12)

    return plt


''' GIF DA VORTICIDADE '''
titulo = 'Vorticidade'
frames_para_animar = frames_vorticidade
anim = animation.FuncAnimation(plt.figure(), mapa_filme, interval=1, frames=len(frames_vorticidade), repeat=False)
anim.save(f'{caminho}{id}vorticidade.gif', writer='pillow', fps=20)


''' GIF DA VELOCIDADE '''
titulo = 'Velocidade'
frames_para_animar = frames_modulo_velocidade
anim = animation.FuncAnimation(plt.figure(), mapa_filme, interval=1, frames=len(frames_modulo_velocidade), repeat=False)
anim.save(f'{caminho}{id}velocidade.gif', writer='pillow', fps=20)

''' GIF DA PRESSAO '''
titulo = 'Pressão'
frames_para_animar = frames_pressao
anim = animation.FuncAnimation(plt.figure(), mapa_filme, interval=1, frames=len(frames_pressao), repeat=False)
anim.save(f'{caminho}{id}pressao.gif', writer='pillow', fps=20)

# plt.quiver(x, y, np.transpose(frames_u[k]), np.transpose(frames_v[k]))
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
