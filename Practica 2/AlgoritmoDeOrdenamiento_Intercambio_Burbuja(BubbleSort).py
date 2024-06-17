# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 20:47:35 2024

@author: jared Montes Candela

Mecatronica 6-E
"""

def bubble_sort(arr):
    """
    Ordena una lista de elementos utilizando el algoritmo de ordenamiento de burbuja.

    Parámetros:
    arr (list): Lista de elementos a ordenar.

    Retorna:
    list: Lista de elementos ordenados.
    """
    n = len(arr)
    # Recorre toda la lista
    for i in range(n):
        # Bandera para verificar si hubo intercambios en la pasada actual
        swapped = False
        # Últimos i elementos ya están ordenados
        for j in range(0, n - i - 1):
            # Si el elemento actual es mayor que el siguiente, intercambiarlos
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # Si no hubo intercambios, la lista ya está ordenada
        if not swapped:
            break
    return arr

# Ejemplo de uso:
if __name__ == "__main__":
    # Lista desordenada de ejemplo
    unsorted_list = [23, 1, 45, 78, 12, 89, 33, 2]

    # Ordena la lista utilizando el algoritmo de ordenamiento de burbuja
    sorted_list = bubble_sort(unsorted_list)

    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_list)
