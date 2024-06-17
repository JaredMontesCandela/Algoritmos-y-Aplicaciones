# -*- coding: utf-8 -*-
"""


@author: jared Montes Candela
"""

import heapq
import networkx as nx
import matplotlib.pyplot as plt

def dijkstra(graph, start):
    """
    Implementación del Algoritmo de Dijkstra.
    
    :param graph: Un diccionario donde las claves son los nodos y los valores son listas de tuplas (vecino, peso)
    :param start: El nodo inicial desde el cual calcular las distancias más cortas
    :return: Un diccionario con las distancias más cortas desde el nodo inicial a todos los otros nodos
    """
    # Inicializamos el diccionario de distancias con infinito para todos los nodos
    distances = {node: float('inf') for node in graph}
    # La distancia al nodo inicial es 0
    distances[start] = 0
    
    # Creamos una cola de prioridad para explorar los nodos, inicializada con el nodo de inicio
    priority_queue = [(0, start)]
    
    # Diccionario para almacenar los caminos más cortos
    shortest_paths = {node: [] for node in graph}
    shortest_paths[start] = [start]
    
    while priority_queue:
        # Extraemos el nodo con la distancia mínima
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Si la distancia actual es mayor que la almacenada, continuamos (esto ocurre si el nodo ya fue procesado)
        if current_distance > distances[current_node]:
            continue
        
        # Exploramos los vecinos del nodo actual
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            # Si encontramos una distancia menor, actualizamos y añadimos el vecino a la cola de prioridad
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                # Actualizamos el camino más corto
                shortest_paths[neighbor] = shortest_paths[current_node] + [neighbor]
    
    return distances, shortest_paths

def draw_graph(graph, shortest_paths):
    """
    Dibuja el grafo y los caminos más cortos calculados por el algoritmo de Dijkstra.
    
    :param graph: Un diccionario donde las claves son los nodos y los valores son listas de tuplas (vecino, peso)
    :param shortest_paths: Un diccionario con los caminos más cortos desde el nodo inicial a todos los otros nodos
    """
    G = nx.Graph()
    
    # Añadimos los nodos y las aristas al grafo
    for node in graph:
        for neighbor, weight in graph[node]:
            G.add_edge(node, neighbor, weight=weight)
    
    pos = nx.spring_layout(G)  # Calculamos las posiciones de los nodos
    
    # Dibujamos los nodos y las aristas del grafo
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_weight='bold')
    edge_labels = {(u, v): f'{d["weight"]}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Dibujamos los caminos más cortos
    for path in shortest_paths.values():
        if len(path) > 1:
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', width=2)
    
    plt.title('Caminos más cortos con el Algoritmo de Dijkstra')
    plt.show()

# Definición del grafo como un diccionario
# Cada clave es un nodo y su valor es una lista de tuplas (vecino, peso)
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}

# Llamada a la función de Dijkstra con el nodo inicial 'A'
shortest_distances, shortest_paths = dijkstra(graph, 'A')

# Imprimimos las distancias más cortas desde 'A' a todos los otros nodos
print("Distancias más cortas desde el nodo 'A':")
for node, distance in shortest_distances.items():
    print(f"Distancia a {node}: {distance}")

# Dibujamos el grafo y los caminos más cortos
draw_graph(graph, shortest_paths)
