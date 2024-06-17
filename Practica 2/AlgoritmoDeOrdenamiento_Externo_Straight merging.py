# -*- coding: utf-8 -*-
import os

"""
Created on Sat Jun  8 22:58:12 2024

@author: jared Montes Candela

Mecatronica 6-E
"""
"""
 Ordena un archivo utilizando el algoritmo de Straight Merging (mezcla directa).

 Parámetros:
 input_file (str): Ruta del archivo de entrada desordenado.
 output_file (str): Ruta del archivo de salida ordenado.
 buffer_size (int): Tamaño del búfer en bytes para la lectura y escritura de datos.

 Retorna:
 None
 """
def straight_merging_sort(input_file, output_file, buffer_size):
   
    # Abrir el archivo de entrada para lectura
    with open(input_file, 'r') as file:
        # Leer el archivo en bloques del tamaño del búfer
        buffer = file.read(buffer_size)
        block_number = 0

        # Procesar cada bloque del archivo de entrada
        while buffer:
            # Convertir el bloque en una lista de números enteros
            numbers = list(map(int, buffer.split()))
            numbers.sort()  # Ordenar el bloque

            # Escribir el bloque ordenado en un archivo temporal
            with open(f'temp_block_{block_number}.txt', 'w') as temp_file:
                temp_file.write(' '.join(map(str, numbers)))

            # Leer el siguiente bloque
            buffer = file.read(buffer_size)
            block_number += 1

    # Fusionar los bloques ordenados
    merge_blocks(block_number, output_file)


def merge_blocks(num_blocks, output_file):
    """
    Fusiona los bloques ordenados en un solo archivo ordenado.

    Parámetros:
    num_blocks (int): Número de bloques a fusionar.
    output_file (str): Ruta del archivo de salida ordenado.

    Retorna:
    None
    """
    # Crear una lista de archivos temporales
    temp_files = [open(f'temp_block_{i}.txt', 'r') for i in range(num_blocks)]

    # Abrir el archivo de salida para escritura
    with open(output_file, 'w') as output:
        # Fusionar los bloques
        while temp_files:
            # Obtener el menor elemento de los bloques actuales
            min_line = min(temp_file.readline().strip() for temp_file in temp_files if temp_file)

            # Escribir el menor elemento en el archivo de salida
            if min_line:
                output.write(min_line + '\n')

            # Eliminar el archivo temporal si se ha llegado al final
            for temp_file in temp_files:
                if temp_file and temp_file.readline().strip() == '':
                    temp_file.close()
                    temp_files.remove(temp_file)

    # Eliminar los archivos temporales
    for i in range(num_blocks):
        file_name = f'temp_block_{i}.txt'
        if os.path.exists(file_name):
            os.remove(file_name)


# Ejemplo de uso:
if __name__ == "__main__":
    input_file = 'unsorted_data.txt'  # Archivo de entrada desordenado
    output_file = 'sorted_data.txt'    # Archivo de salida ordenado
    buffer_size = 1024 * 1024           # Tamaño del búfer: 1 MB

    # Llamar a la función straight_merging_sort para ordenar el archivo
    straight_merging_sort(input_file, output_file, buffer_size)
