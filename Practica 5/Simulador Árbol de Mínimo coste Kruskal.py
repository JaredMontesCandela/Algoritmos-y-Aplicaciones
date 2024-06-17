# -*- coding: utf-8 -*-
"""

@author: jared montes candela
"""

import networkx as nx
import matplotlib.pyplot as plt


class UnionFind:
    """ Estructura de datos de Union-Find con compresión de camino. """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Compresión de camino
        return self.parent[u]
    
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        
        if root_u != root_v:
            # Union por rango
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
    
    def connected(self, u, v):
        return self.find(u) == self.find(v)


def kruskal(graph):
    """ Algoritmo de Kruskal para encontrar el árbol de mínimo costo. """
    edges = []
    for u in range(len(graph)):
        for v in range(u + 1, len(graph)):
            if graph[u][v] != float('inf'):  # Asumimos que float('inf') indica ausencia de conexión
                edges.append((graph[u][v], u, v))
    
    edges.sort()  # Ordenar aristas por peso
    
    uf = UnionFind(len(graph))
    mst = []
    
    for weight, u, v in edges:
        if not uf.connected(u, v):
            uf.union(u, v)
            mst.append((u, v, weight))
    
    return mst


# Función para visualizar el grafo y el MST
def visualizar_grafo(graph, mst):
    G = nx.Graph()
    
    # Agregar aristas al grafo
    for u in range(len(graph)):
        for v in range(u + 1, len(graph)):
            if graph[u][v] != float('inf'):
                G.add_edge(u, v, weight=graph[u][v])
    
    # Posiciones de los nodos para una visualización más clara
    pos = nx.spring_layout(G)
    
    # Dibujar el grafo original
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=12, font_weight='bold', edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)})
    plt.title('Grafo Original')
    plt.show()
    
    # Crear grafo para el MST
    MST = nx.Graph()
    MST.add_edges_from([(u, v) for u, v, weight in mst])
    
    # Dibujar el MST
    plt.figure(figsize=(12, 8))
    nx.draw(MST, pos, with_labels=True, node_color='lightgreen', node_size=1500, font_size=12, font_weight='bold', edge_color='green')
    nx.draw_networkx_edge_labels(MST, pos, edge_labels={(u, v): f"{weight}" for u, v, weight in mst})
    plt.title('Árbol de Mínimo Costo (Kruskal)')
    plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de un grafo representado como matriz de adyacencia
    # Aquí asumimos un grafo no dirigido y ponderado, donde inf significa que no hay conexión directa
    graph = [
        [float('inf'), 2, 4, float('inf'), float('inf'), float('inf')],
        [2, float('inf'), 3, 5, 6, float('inf')],
        [4, 3, float('inf'), 1, float('inf'), float('inf')],
        [float('inf'), 5, 1, float('inf'), 7, 8],
        [float('inf'), 6, float('inf'), 7, float('inf'), 9],
        [float('inf'), float('inf'), float('inf'), 8, 9, float('inf')]
    ]
    
    # Obtener el árbol de mínimo costo usando Kruskal
    minimum_spanning_tree = kruskal(graph)
    
    # Visualizar el grafo original y el MST
    visualizar_grafo(graph, minimum_spanning_tree)
