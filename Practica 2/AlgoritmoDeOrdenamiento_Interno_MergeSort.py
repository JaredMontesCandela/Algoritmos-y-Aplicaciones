# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 22:44:00 2024

@author: jared Montes Candela

Mecatronica 6-E
"""

def merge_sort(arr):
    """
    Ordenamiento por Mezcla (MergeSort).
    
    Objetivo: Ordenar una lista de elementos de manera ascendente
    utilizando el método de ordenamiento por mezcla.
    
    Parámetros:
    arr (list): Lista de elementos a ordenar.
    
    Retorna:
    list: Lista ordenada de elementos.
    """
    
    # Caso base: Si la lista tiene 0 o 1 elementos, ya está ordenada
    if len(arr) <= 1:
        return arr

    # Divide la lista en dos mitades
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Llama recursivamente a merge_sort en ambas mitades
    left_sorted = merge_sort(left_half)
    right_sorted = merge_sort(right_half)

    # Mezcla las dos mitades ordenadas
    return merge(left_sorted, right_sorted)

def merge(left, right):
    """
    Mezcla dos listas ordenadas en una sola lista ordenada.
    
    Parámetros:
    left (list): Primera lista ordenada.
    right (list): Segunda lista ordenada.
    
    Retorna:
    list: Lista mezclada y ordenada.
    """
    
    sorted_list = []  # Lista para almacenar el resultado ordenado
    i = j = 0  # Punteros para las listas left y right
    
    # Compara elementos de ambas listas y añádelos en orden a sorted_list
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            sorted_list.append(left[i])
            i += 1
        else:
            sorted_list.append(right[j])
            j += 1

    # Añade los elementos restantes de left, si los hay
    while i < len(left):
        sorted_list.append(left[i])
        i += 1

    # Añade los elementos restantes de right, si los hay
    while j < len(right):
        sorted_list.append(right[j])
        j += 1

    return sorted_list

# Ejemplo de uso:
if __name__ == "__main__":
    # Lista de ejemplo desordenada
    arr = [38, 27, 43, 3, 9, 82, 10]
    
    # Imprime la lista original
    print("Lista original:", arr)
    
    # Ordena la lista utilizando el método de ordenamiento por mezcla (MergeSort)
    sorted_arr = merge_sort(arr)
    
    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_arr)


