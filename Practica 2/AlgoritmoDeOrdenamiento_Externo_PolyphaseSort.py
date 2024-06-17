# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:30:15 2024

@author: jared Montes Candela

Mecatronica 6-E
"""
import os
import heapq

def split_into_runs(input_file, buffer_size):
    """
    Divide el archivo de entrada en secuencias ordenadas (runs) y las escribe en archivos temporales.

    Parámetros:
    input_file (str): Ruta del archivo de entrada desordenado.
    buffer_size (int): Tamaño del búfer en bytes para la lectura de datos.

    Retorna:
    list: Lista de rutas de archivos temporales con secuencias ordenadas.
    """
    run_files = []
    run_number = 0

    with open(input_file, 'r') as file:
        buffer = file.read(buffer_size)

        while buffer:
            numbers = list(map(int, buffer.split()))
            numbers.sort()

            run_file = f'run_file_{run_number}.txt'
            run_files.append(run_file)

            with open(run_file, 'w') as temp_file:
                for number in numbers:
                    temp_file.write(f'{number}\n')

            buffer = file.read(buffer_size)
            run_number += 1

    return run_files

def polyphase_merge(run_files, output_file):
    """
    Realiza la mezcla polifásica de las secuencias ordenadas y escribe el resultado en el archivo de salida.

    Parámetros:
    run_files (list): Lista de rutas de archivos temporales con secuencias ordenadas.
    output_file (str): Ruta del archivo de salida ordenado.

    Retorna:
    None
    """
    min_heap = []
    file_pointers = []

    # Abrir todos los archivos de run y agregar el primer número de cada archivo al heap
    for i, run_file in enumerate(run_files):
        file_pointer = open(run_file, 'r')
        file_pointers.append(file_pointer)
        first_number = file_pointer.readline().strip()
        if first_number:
            heapq.heappush(min_heap, (int(first_number), i))

    with open(output_file, 'w') as output:
        while min_heap:
            min_value, file_index = heapq.heappop(min_heap)
            output.write(str(min_value) + '\n')

            next_number = file_pointers[file_index].readline().strip()
            if next_number:
                heapq.heappush(min_heap, (int(next_number), file_index))

    # Cerrar todos los archivos de run y eliminarlos
    for file_pointer in file_pointers:
        file_pointer.close()
        os.remove(file_pointer.name)

# Ejemplo de uso:
if __name__ == "__main__":
    input_file = 'unsorted.3_data.txt'
    output_file = 'sorted.3_data.txt'
    buffer_size = 32  # Tamaño del búfer en bytes (ajusta según sea necesario)

    # Divide el archivo de entrada en secuencias ordenadas (runs)
    run_files = split_into_runs(input_file, buffer_size)

    # Realiza la mezcla polifásica para obtener el archivo de salida ordenado
    polyphase_merge(run_files, output_file)
