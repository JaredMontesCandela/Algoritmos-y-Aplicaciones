# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 19:16:41 2024

@author: Jared Montes Candela

Mecatronica 6-E
"""

"""
Ordenamiento por Inserción (Insertion Sort).

Objetivo: Ordenar una lista de elementos de manera ascendente
utilizando el método de inserción.

Parámetros:
arr (list): Lista de elementos a ordenar.

Retorna:
list: Lista ordenada de elementos.
"""

def insertion_sort(arr):
    
    
    # Itera a través de cada elemento de la lista, comenzando desde el segundo elemento
    for i in range(1, len(arr)):
        key = arr[i]  # El elemento a insertar en la parte ordenada
        j = i - 1     # El índice del elemento anterior en la lista ordenada

        # Mueve los elementos de arr[0..i-1], que son mayores que la key,
        # una posición adelante para crear espacio para la key
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]  # Mueve el elemento hacia adelante
            j -= 1  # Mueve el índice hacia atrás

        # Inserta la key en la posición correcta
        arr[j + 1] = key

    return arr  # Retorna la lista ordenada

# Ejemplo de uso:
if __name__ == "__main__":
    # Lista de ejemplo desordenada
    arr = [12, 11, 13, 5, 6]
    
    # Imprime la lista original
    print("Lista original:", arr)
    
    # Ordena la lista utilizando el método de inserción
    sorted_arr = insertion_sort(arr)
    
    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_arr)
