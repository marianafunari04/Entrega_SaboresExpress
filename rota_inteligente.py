import heapq
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Grafo: cada ponto e suas conexões (vizinhos) com peso
# (Exemplo simples, personalize para sua cidade)
grafo = {
    'A': {'B': 2, 'C': 5},
    'B': {'A': 2, 'D': 3},
    'C': {'A': 5, 'D': 2, 'E': 3},
    'D': {'B': 3, 'C': 2, 'E': 4},
    'E': {'C': 3, 'D': 4}
}

# Coordenadas hipotéticas para visualização (x, y)
coordenadas = {
    'A': (1, 1),
    'B': (2, 4),
    'C': (5, 2),
    'D': (4, 5),
    'E': (7, 5)
}

# Algoritmo A* para caminho mais curto entre dois pontos
def heuristica(a, b):
    x1, y1 = coordenadas[a]
    x2, y2 = coordenadas[b]
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def a_estrela(grafo, inicio, objetivo):
    fila = []
    heapq.heappush(fila, (0, inicio))
    caminho = {inicio: None}
    custo = {inicio: 0}
    while fila:
        _, atual = heapq.heappop(fila)
        if atual == objetivo:
            break
        for viz, peso in grafo[atual].items():
            novo_custo = custo[atual] + peso
            if viz not in custo or novo_custo < custo[viz]:
                custo[viz] = novo_custo
                prioridade = novo_custo + heuristica(viz, objetivo)
                heapq.heappush(fila, (prioridade, viz))
                caminho[viz] = atual
    # Reconstrução do caminho
    if objetivo not in caminho:
        return None
    rota = []
    atual = objetivo
    while atual:
        rota.append(atual)
        atual = caminho[atual]
    rota.reverse()
    return rota

# K-Means para agrupar entregas

def agrupar_entregas(coordenadas, n_clusters=2):
    pontos = np.array(list(coordenadas.values()))
    modelo = KMeans(n_clusters=n_clusters, random_state=42)
    modelo.fit(pontos)
    return modelo.labels_, modelo.cluster_centers_

# Visualização dos clusters
def visualizar_clusters(coordenadas, labels, centers):
    pontos = np.array(list(coordenadas.values()))
    plt.figure(figsize=(7,5))
    plt.scatter(pontos[:, 0], pontos[:, 1], c=labels, cmap='viridis', s=100, label='Pontos de entrega')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centros dos clusters')
    for i, nome in enumerate(coordenadas.keys()):
        plt.annotate(nome, (pontos[i, 0]+0.1, pontos[i, 1]+0.1), fontsize=12, color='black')
    plt.title('Agrupamento de entregas (K-Means)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Ajuste a quantidade de clusters conforme o número de pedidos real:
    n_clusters = 2
    
    # Agrupa as entregas
    labels, centers = agrupar_entregas(coordenadas, n_clusters)
    print(f'Labels dos clusters: {labels}')
    print(f'Centros: {centers}')
    visualizar_clusters(coordenadas, labels, centers)
    
    # Exemplo: buscar menor rota de A até E
    rota = a_estrela(grafo, 'A', 'E')
    if rota:
        print(f'Rota otimizada de A até E: {" -> ".join(rota)}')
    else:
        print('Não foi encontrada rota entre A e E.')
