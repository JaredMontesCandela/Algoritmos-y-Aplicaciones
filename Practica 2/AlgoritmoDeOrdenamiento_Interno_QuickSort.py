# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 22:37:04 2024

@author: jared Montes Candela

Mecatronica 6-E
"""

"""
 Ordenamiento Rápido (QuickSort).
 
 Objetivo: Ordenar una lista de elementos de manera ascendente
 utilizando el método de ordenamiento rápido.
 
 Parámetros:
 arr (list): Lista de elementos a ordenar.
 
 Retorna:
 list: Lista ordenada de elementos.
 """
def quick_sort(arr):

    
    # Caso base: Si la lista tiene 0 o 1 elementos, ya está ordenada
    if len(arr) <= 1:
        return arr

    # Elección del pivote: Utilizamos el último elemento de la lista como pivote
    pivot = arr[len(arr) - 1]
    
    # Particiones: 
    left = [x for x in arr[:-1] if x <= pivot]  # Elementos menores o iguales al pivote
    right = [x for x in arr[:-1] if x > pivot]  # Elementos mayores al pivote

    # Recursión: Aplicamos QuickSort en las sublistas izquierda y derecha
    return quick_sort(left) + [pivot] + quick_sort(right)

# Ejemplo de uso:
if __name__ == "__main__":
    # Lista de ejemplo desordenada
    arr = [10, 7, 8, 9, 1, 5]
    
    # Imprime la lista original
    print("Lista original:", arr)
    
    # Ordena la lista utilizando el método de ordenamiento rápido (QuickSort)
    sorted_arr = quick_sort(arr)
    
    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_arr)
