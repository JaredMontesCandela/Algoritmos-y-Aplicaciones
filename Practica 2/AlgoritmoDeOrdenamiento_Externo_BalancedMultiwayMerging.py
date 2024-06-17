# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:12:42 2024

@author: jared Montes Candela

Mecatronica 6-E
"""

import heapq
import os
"""
Ordena múltiples archivos utilizando el algoritmo de Balanced Multiway Merging.

Parámetros:
input_files (list): Lista de rutas de archivos de entrada desordenados.
output_file (str): Ruta del archivo de salida ordenado.

Retorna:
None
"""
def balanced_multiway_merging(input_files, output_file):
    
    # Abrir todos los archivos de entrada
    input_file_pointers = [open(file, 'r') for file in input_files]

    # Crear una lista de heap para fusionar los elementos de todos los archivos
    heap = []

    # Leer el primer elemento de cada archivo y agregarlo al heap
    for file_pointer in input_file_pointers:
        element = file_pointer.readline().strip()
        if element:
            heapq.heappush(heap, (int(element), file_pointer))

    # Abrir el archivo de salida para escritura
    with open(output_file, 'w') as output:
        # Fusionar los elementos de los archivos hasta que el heap esté vacío
        while heap:
            # Obtener el elemento mínimo del heap (menor elemento de todos los archivos)
            min_element, min_file_pointer = heapq.heappop(heap)

            # Escribir el elemento mínimo en el archivo de salida
            output.write(str(min_element) + '\n')

            # Leer el siguiente elemento del archivo del que se extrajo el elemento mínimo
            next_element = min_file_pointer.readline().strip()

            # Si hay un siguiente elemento, agregarlo al heap
            if next_element:
                heapq.heappush(heap, (int(next_element), min_file_pointer))
            # Si no hay más elementos en el archivo, cerrarlo
            else:
                min_file_pointer.close()

    # Cerrar todos los archivos de entrada
    for file_pointer in input_file_pointers:
        file_pointer.close()


# Ejemplo de uso:
if __name__ == "__main__":
    input_files = ['input_file1.txt', 'input_file2.txt', 'input_file3.txt']
    output_file = 'sorted_files_data.txt'

    # Llamar a la función balanced_multiway_merging para ordenar los archivos de entrada
    balanced_multiway_merging(input_files, output_file)
