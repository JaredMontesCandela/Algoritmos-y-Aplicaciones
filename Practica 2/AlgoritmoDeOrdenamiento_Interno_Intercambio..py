# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 22:29:03 2024

@author: jared Montes Candela

Mecatronica 6-E
"""
"""
Ordenamiento por Intercambio (Bubble Sort).

Objetivo: Ordenar una lista de elementos de manera ascendente
utilizando el método de intercambio.

Parámetros:
arr (list): Lista de elementos a ordenar.

Retorna:
list: Lista ordenada de elementos.
"""

def bubble_sort(arr):
    
    
    n = len(arr)  # Longitud de la lista
    
    # Itera a través de todos los elementos de la lista
    for i in range(n):
        # Bucle interno para el intercambio
        for j in range(0, n-i-1):
            # Si el elemento actual es mayor que el siguiente, intercambiarlos
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    
    return arr  # Retorna la lista ordenada

# Ejemplo de uso:
if __name__ == "__main__":
    # Lista de ejemplo desordenada
    arr = [64, 34, 25, 12, 22, 11, 90]
    
    # Imprime la lista original
    print("Lista original:", arr)
    
    # Ordena la lista utilizando el método de intercambio (Bubble Sort)
    sorted_arr = bubble_sort(arr)
    
    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_arr)

