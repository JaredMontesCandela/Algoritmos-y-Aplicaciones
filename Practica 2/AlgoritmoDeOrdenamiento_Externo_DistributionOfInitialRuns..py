# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:50:28 2024

@author: jared Montes Candela

Mecatronica 6-E
"""
"""
  Divide el archivo de entrada en secuencias ordenadas (runs) y las escribe en archivos temporales.

  Parámetros:
  input_file (str): Ruta del archivo de entrada desordenado.
  output_dir (str): Directorio donde se guardarán los archivos temporales.
  buffer_size (int): Tamaño del búfer en bytes para la lectura de datos.

  Retorna:
  list: Lista de rutas de archivos temporales con secuencias ordenadas.
  """
import os

def split_into_initial_runs(input_file, output_dir, buffer_size):
  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    run_files = []
    run_number = 0

    with open(input_file, 'r') as file:
        buffer = file.read(buffer_size)

        while buffer:
            numbers = list(map(int, buffer.split()))
            numbers.sort()

            run_file = os.path.join(output_dir, f'run_file.2_{run_number}.txt')
            run_files.append(run_file)

            with open(run_file, 'w') as temp_file:
                for number in numbers:
                    temp_file.write(f'{number}\n')

            buffer = file.read(buffer_size)
            run_number += 1

    return run_files

# Ejemplo de uso:
if __name__ == "__main__":
    # Ruta del archivo de entrada desordenado
    input_file = 'unsorted.4_data.txt'
    # Directorio donde se guardarán los archivos temporales
    output_dir = 'runs'
    # Tamaño del búfer en bytes (ajusta según sea necesario)
    buffer_size = 32

    # Divide el archivo de entrada en secuencias ordenadas (runs)
    run_files = split_into_initial_runs(input_file, output_dir, buffer_size)

    print("Runs iniciales creadas:")
    for run_file in run_files:
        print(run_file)
