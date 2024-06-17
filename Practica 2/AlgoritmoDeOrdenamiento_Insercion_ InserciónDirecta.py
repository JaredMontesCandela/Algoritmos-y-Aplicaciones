# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 20:09:49 2024

@author: jared Montes Candela

Mecatronica 6-E
"""
"""
Ordena una lista de elementos utilizando el algoritmo de ordenamiento por inserción directa.

Parámetros:
arr (list): Lista de elementos a ordenar.

Retorna:
list: Lista de elementos ordenados.
"""

def insertion_sort(arr):
    
    # Recorre cada elemento en la lista, empezando desde el segundo elemento
    for i in range(1, len(arr)):
        key = arr[i]  # El elemento actual a ser insertado en la parte ordenada de la lista
        j = i - 1

        # Mueve los elementos de la parte ordenada de la lista que son mayores que 'key' una posición adelante
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        # Inserta 'key' en su posición correcta
        arr[j + 1] = key

    return arr

# Ejemplo de uso:
if __name__ == "__main__":
    # Lista desordenada de ejemplo
    unsorted_list = [23, 1, 45, 78, 12, 89, 33, 2]

    # Ordena la lista utilizando el algoritmo de ordenamiento por inserción directa
    sorted_list = insertion_sort(unsorted_list)

    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_list)
