# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 19:31:55 2024

@author: jared Montes Candela

Mecatronica 6-E
"""
  
"""
  Ordenamiento por Selección (Selection Sort).
  
  Objetivo: Ordenar una lista de elementos de manera ascendente
  utilizando el método de selección.
  
  Parámetros:
  arr (list): Lista de elementos a ordenar.
  
  Retorna:
  list: Lista ordenada de elementos.
  """
  
def selection_sort(arr):
  
    n = len(arr)  # Longitud de la lista
    
    # Itera a través de cada elemento de la lista
    for i in range(n):
        # Encuentra el índice del elemento mínimo en la sublista arr[i..n-1]
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Intercambia el elemento encontrado con el primer elemento de la sublista
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr  # Retorna la lista ordenada

# Ejemplo de uso:
if __name__ == "__main__":
    # Lista de ejemplo desordenada
    arr = [64, 25, 12, 22, 11]
    
    # Imprime la lista original
    print("Lista original:", arr)
    
    # Ordena la lista utilizando el método de selección
    sorted_arr = selection_sort(arr)
    
    # Imprime la lista ordenada
    print("Lista ordenada:", sorted_arr)
