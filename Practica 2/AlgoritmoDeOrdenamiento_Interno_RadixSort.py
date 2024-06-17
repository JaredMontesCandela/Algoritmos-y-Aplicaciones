# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 22:57:02 2024

@author: jared Montes Candela

Mecatronica 6-E
"""
"""
Ordena la lista utilizando el algoritmo de Counting Sort basado en el dígito específico.

Parámetros:
arr (list): Lista de elementos a ordenar.
exp (int): El exponente que representa el dígito específico en el que se está ordenando.

Retorna:
list: Lista ordenada de elementos.
"""

def counting_sort(arr, exp):
    
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    # Cuenta la frecuencia de cada dígito
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    # Calcula la posición actual de cada dígito en el array de salida
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Construye el array de salida
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    # Copia el array de salida al array original
    for i in range(n):
        arr[i] = output[i]


def radix_sort(arr):
    """
    Ordenamiento Radix Sort.

    Parámetros:
    arr (list): Lista de elementos a ordenar.

    Retorna:
    list: Lista ordenada de elementos.
    """
    # Encuentra el número máximo para determinar el número de dígitos
    max_element = max(arr)

    # Aplica Counting Sort para cada dígito, comenzando desde el dígito menos significativo
    exp = 1
    while max_element // exp > 0:
        counting_sort(arr, exp)
        exp *= 10

    return arr


# Ejemplo de uso:
if __name__ == "__main__":
    # Lista de ejemplo desordenada
    arr = [170, 45, 75, 90, 802, 24, 2, 66]
    
    # Imprime la lista original
    print("Lista original:", arr)
    
    # Ordena la lista utilizando el método de ordenamiento Radix Sort
    sorted_arr = radix_sort(arr)
    
    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_arr)

