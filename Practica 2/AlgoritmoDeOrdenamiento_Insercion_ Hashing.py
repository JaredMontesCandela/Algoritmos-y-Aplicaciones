# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 20:45:38 2024

@author: jared Montes Candela

Mecatronica 6-E
"""

def hash_function(value, num_buckets):
    
    return value % num_buckets

def insertion_sort(arr):
    """
    Ordena una lista de elementos utilizando el algoritmo de ordenamiento por inserción directa.

    Parámetros:
    arr (list): Lista de elementos a ordenar.

    Retorna:
    list: Lista de elementos ordenados.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def hash_sort(arr, num_buckets):
    """
    Ordena una lista de elementos utilizando el algoritmo de ordenamiento por hashing.

    Parámetros:
    arr (list): Lista de elementos a ordenar.
    num_buckets (int): Número de cubetas a utilizar.

    Retorna:
    list: Lista de elementos ordenados.
    """
    # Crear cubetas vacías
    buckets = [[] for _ in range(num_buckets)]

    # Distribuir los elementos en las cubetas
    for value in arr:
        bucket_index = hash_function(value, num_buckets)
        buckets[bucket_index].append(value)

    # Ordenar cada cubeta y combinar los resultados
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(insertion_sort(bucket))

    return sorted_arr

# Ejemplo de uso:
if __name__ == "__main__":
    # Lista desordenada de ejemplo
    unsorted_list = [23, 1, 45, 78, 12, 89, 33, 2]

    # Número de cubetas
    num_buckets = 5

    # Ordena la lista utilizando el algoritmo de ordenamiento por hashing
    sorted_list = hash_sort(unsorted_list, num_buckets)

    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_list)
