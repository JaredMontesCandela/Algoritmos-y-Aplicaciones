# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 20:14:07 2024

@author: jared Montes Candela

Mecatronica 6-E
"""

def binary_search(arr, val, start, end):
    """
Realiza una búsqueda binaria para encontrar la posición correcta de 'val' en 'arr'.

Parámetros:
arr (list): Lista de elementos.
val (int): Valor a insertar.
start (int): Índice inicial de la sublista para la búsqueda.
end (int): Índice final de la sublista para la búsqueda.

Retorna:
int: Índice donde 'val' debe ser insertado.
"""    
    if start == end:
        if arr[start] > val:
            return start
        else:
            return start + 1

    if start > end:
        return start

    mid = (start + end) // 2
    if arr[mid] < val:
        return binary_search(arr, val, mid + 1, end)
    elif arr[mid] > val:
        return binary_search(arr, val, start, mid - 1)
    else:
        return mid

def binary_insertion_sort(arr):
    """
    Ordena una lista de elementos utilizando el algoritmo de inserción binaria.

    Parámetros:
    arr (list): Lista de elementos a ordenar.

    Retorna:
    list: Lista de elementos ordenados.
    """
    for i in range(1, len(arr)):
        val = arr[i]
        # Encuentra la posición correcta para 'val' usando búsqueda binaria
        j = binary_search(arr, val, 0, i - 1)
        # Mueve los elementos para hacer espacio para 'val'
        arr = arr[:j] + [val] + arr[j:i] + arr[i+1:]

    return arr

# Ejemplo de uso:
if __name__ == "__main__":
    # Lista desordenada de ejemplo
    unsorted_list = [23, 1, 45, 78, 12, 89, 33, 2]

    # Ordena la lista utilizando el algoritmo de inserción binaria
    sorted_list = binary_insertion_sort(unsorted_list)

    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_list)
