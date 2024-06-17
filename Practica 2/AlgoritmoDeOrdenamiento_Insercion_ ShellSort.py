# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 20:10:58 2024

@author: jared Montes Candela

Mecatronica 6-E
"""
"""
Ordena una lista de elementos utilizando el algoritmo ShellSort.

Parámetros:
arr (list): Lista de elementos a ordenar.

Retorna:
list: Lista de elementos ordenados.
"""
def shell_sort(arr):
    
    n = len(arr)
    gap = n // 2  # Inicialmente, el gap es la mitad del tamaño de la lista

    # Reduce el gap y realiza inserciones directas para cada gap
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]  # El elemento actual a ser insertado en la parte ordenada de la lista
            j = i
            # Mueve los elementos de la parte ordenada de la lista que son mayores que 'temp' una posición adelante
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap

            # Inserta 'temp' en su posición correcta
            arr[j] = temp

        # Reduce el gap para la siguiente iteración
        gap //= 2

    return arr

# Ejemplo de uso:
if __name__ == "__main__":
    # Lista desordenada de ejemplo
    unsorted_list = [23, 1, 45, 78, 12, 89, 33, 2]

    # Ordena la lista utilizando el algoritmo ShellSort
    sorted_list = shell_sort(unsorted_list)

    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_list)
