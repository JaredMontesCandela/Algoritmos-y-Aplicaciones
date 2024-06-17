# -*- coding: utf-8 -*-
"""

@author: jared Montes Candela
"""

import heapq
import networkx as nx
import matplotlib.pyplot as plt

def prim_mst(graph):
    """
    Función para encontrar el Árbol de Expansión Mínima (MST) de un grafo usando el algoritmo de Prim.
    :param graph: Diccionario donde las llaves son los nodos y los valores son listas de tuplas (vecino, peso)
    :return: Lista de aristas (u, v, peso) que componen el MST
    """

    # Inicializamos el nodo inicial (en este caso, el primer nodo del grafo)
    start_node = next(iter(graph))
    
    # Priority queue (cola de prioridad) para seleccionar la arista de menor peso
    pq = []
    # Agregamos todas las aristas del nodo inicial a la cola de prioridad
    for neighbor, weight in graph[start_node]:
        heapq.heappush(pq, (weight, start_node, neighbor))

    # Conjunto para mantener un seguimiento de los nodos que ya están en el MST
    in_mst = set([start_node])
    
    # Lista para almacenar las aristas del MST
    mst_edges = []
    
    # Mientras la cola de prioridad no esté vacía y no hayamos incluido todos los nodos en el MST
    while pq and len(in_mst) < len(graph):
        # Sacamos la arista de menor peso de la cola de prioridad
        weight, u, v = heapq.heappop(pq)
        
        # Si el nodo destino ya está en el MST, continuamos
        if v in in_mst:
            continue
        
        # Añadimos la arista al MST
        mst_edges.append((u, v, weight))
        # Añadimos el nuevo nodo al conjunto de nodos en el MST
        in_mst.add(v)
        
        # Añadimos todas las aristas del nuevo nodo a la cola de prioridad
        for neighbor, weight in graph[v]:
            if neighbor not in in_mst:
                heapq.heappush(pq, (weight, v, neighbor))
    
    return mst_edges

# Ejemplo de grafo representado como un diccionario de listas de adyacencia
graph = {
    'A': [('B', 7), ('D', 5)],
    'B': [('A', 7), ('C', 8), ('D', 9), ('E', 7)],
    'C': [('B', 8), ('E', 5)],
    'D': [('A', 5), ('B', 9), ('E', 15), ('F', 6)],
    'E': [('B', 7), ('C', 5), ('D', 15), ('F', 8), ('G', 9)],
    'F': [('D', 6), ('E', 8), ('G', 11)],
    'G': [('E', 9), ('F', 11)]
}

# Ejecutar el algoritmo de Prim para encontrar el MST
mst = prim_mst(graph)

# Imprimir el resultado
print("El Árbol de Expansión Mínima (MST) consiste en las siguientes aristas:")
for u, v, weight in mst:
    print(f"{u} -- {v} (peso: {weight})")

# Visualización gráfica del grafo y su MST
G = nx.Graph()

# Añadir todas las aristas al grafo
for node in graph:
    for neighbor, weight in graph[node]:
        G.add_edge(node, neighbor, weight=weight)

# Dibujar el grafo completo
pos = nx.spring_layout(G)  # Posiciones de los nodos

plt.figure(figsize=(10, 8))

# Dibujar todas las aristas (grafo completo) en gris
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
# Dibujar las aristas del MST en azul
mst_edges_tuples = [(u, v) for u, v, weight in mst]
nx.draw_networkx_edges(G, pos, edgelist=mst_edges_tuples, edge_color='blue', width=2)

# Dibujar nodos y etiquetas
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

# Dibujar pesos de las aristas
edge_labels = {(u, v): weight for u, v, weight in mst}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Árbol de Expansión Mínima usando el algoritmo de Prim")
plt.show()
