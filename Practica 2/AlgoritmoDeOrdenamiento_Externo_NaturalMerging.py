# -*- coding: utf-8 -*-
import os

"""
Created on Sat Jun  8 23:03:45 2024

@author: jared Montes Candela

Mecatronica 6-E
"""

"""
 Ordena un archivo utilizando el algoritmo de Natural Merging (mezcla natural).

 Parámetros:
 input_file (str): Ruta del archivo de entrada desordenado.
 output_file (str): Ruta del archivo de salida ordenado.
 buffer_size (int): Tamaño del búfer en bytes para la lectura y escritura de datos.

 Retorna:
 None
 """
def natural_merging_sort(input_file, output_file, buffer_size):
  
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

    # Fusionar los bloques ordenados de manera natural
    natural_merge_blocks(block_number, output_file)


def natural_merge_blocks(num_blocks, output_file):
    """
    Fusiona los bloques ordenados de manera natural en un solo archivo ordenado.

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
        # Fusionar los bloques de manera natural
        while temp_files:
            # Fusionar bloques consecutivos que ya están ordenados
            consecutive_blocks = []
            current_block = []

            while temp_files:
                # Leer la primera línea del bloque actual
                line = temp_files[0].readline().strip()

                if line:
                    # Convertir la línea en una lista de números
                    numbers = list(map(int, line.split()))
                    
                    if not current_block or numbers[0] >= current_block[-1]:
                        # Agregar el bloque al grupo de bloques consecutivos ordenados
                        consecutive_blocks.append(numbers)
                        current_block = numbers
                    else:
                        # Si el bloque no está ordenado, detener la fusión
                        break
                else:
                    # Si el bloque está vacío, cerrar el archivo y eliminarlo de la lista
                    temp_files[0].close()
                    temp_files.pop(0)

            # Fusionar bloques consecutivos ordenados
            merged_block = merge_consecutive_blocks(consecutive_blocks)

            # Escribir el bloque fusionado en el archivo de salida
            output.write(' '.join(map(str, merged_block)) + '\n')

    # Eliminar los archivos temporales
    for i in range(num_blocks):
        file_name = f'temp_block_{i}.txt'
        if os.path.exists(file_name):
            os.remove(file_name)


def merge_consecutive_blocks(consecutive_blocks):
    """
    Fusiona bloques consecutivos ordenados en un solo bloque ordenado.

    Parámetros:
    consecutive_blocks (list): Lista de bloques consecutivos ordenados.

    Retorna:
    list: Bloque fusionado y ordenado.
    """
    merged_block = []
    iterators = [iter(block) for block in consecutive_blocks]
    current_values = [next(iterator, None) for iterator in iterators]

    while any(current_values):
        # Obtener el valor mínimo actual de todos los bloques
        min_value = min(filter(None, current_values))

        # Agregar el valor mínimo al bloque fusionado
        merged_block.append(min_value)

        # Actualizar los valores actuales de los bloques
        current_values = [next(iterator, None) for iterator, value in zip(iterators, current_values) if value == min_value]

    return merged_block


# Ejemplo de uso:
if __name__ == "__main__":
    input_file = 'unsorted.2_data.txt'  # Archivo de entrada desordenado
    output_file = 'sorted.2_data.txt'    # Archivo de salida ordenado
    buffer_size = 1024 * 1024           # Tamaño del búfer: 1 MB

    # Llamar a la función natural_merging_sort para ordenar el archivo
    natural_merging_sort(input_file, output_file, buffer_size)
