# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 22:32:21 2024

@author: jared Montes Candela

Mecatronica 6-E
"""

"""
 Función para convertir un subárbol en un montón (heap).
 
 Objetivo: Asegurar que el subárbol con raíz en el índice 'i' cumpla
 con la propiedad del montón.
 
 Parámetros:
 arr (list): Lista de elementos a ordenar.
 n (int): Tamaño del montón.
 i (int): Índice del nodo raíz del subárbol.
 """

def heapify(arr, n, i):
   
    
    largest = i  # Inicializa el nodo raíz como el mayor
    left = 2 * i + 1  # Hijo izquierdo
    right = 2 * i + 2  # Hijo derecho

    # Si el hijo izquierdo es mayor que la raíz
    if left < n and arr[left] > arr[largest]:
        largest = left

    # Si el hijo derecho es mayor que el mayor hasta ahora
    if right < n and arr[right] > arr[largest]:
        largest = right

    # Si el mayor no es la raíz
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Intercambia la raíz con el mayor
        heapify(arr, n, largest)  # Recursivamente aplica heapify en el subárbol afectado

def heap_sort(arr):
    """
    Ordenamiento de árbol (Heap Sort).
    
    Objetivo: Ordenar una lista de elementos de manera ascendente
    utilizando el método de ordenamiento por montón (Heap Sort).
    
    Parámetros:
    arr (list): Lista de elementos a ordenar.
    
    Retorna:
    list: Lista ordenada de elementos.
    """
    
    n = len(arr)  # Longitud de la lista

    # Construye un montón máximo a partir de la lista
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extrae elementos del montón uno por uno
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # Mueve la raíz actual al final
        heapify(arr, i, 0)  # Llama a heapify en el montón reducido
    
    return arr  # Retorna la lista ordenada

# Ejemplo de uso:
if __name__ == "__main__":
    # Lista de ejemplo desordenada
    arr = [12, 11, 13, 5, 6, 7]
    
    # Imprime la lista original
    print("Lista original:", arr)
    
    # Ordena la lista utilizando el método de ordenamiento por montón (Heap Sort)
    sorted_arr = heap_sort(arr)
    
    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_arr)
