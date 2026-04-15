import sqlite3
import numpy as np
from tabulate import tabulate
import os
import csv

###########################################################
###########################################################

db = 'faces.sqlite'

with sqlite3.connect(db) as conn:
    cur = conn.cursor()
    cur.execute("SELECT filename, embedding FROM testing")
    data = cur.fetchall()

# Готовим имена и embedding'и
filenames = []
embeddings = []
for fname, emb_blob in data:
    # Обрезаем .png (если есть)
    name = fname[:-4] if fname.lower().endswith('.png') else fname
    filenames.append(name)
    embeddings.append(np.frombuffer(emb_blob, dtype=np.float32))

# Строим попарную матрицу расстояний (L2)
n = len(filenames)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        distance_matrix[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])

########################################################################
########################################################################

SHORT_NAMES = 0

if(SHORT_NAMES):
    
    filenames = [name[:2] + name[-2:] for name in filenames]

TABULATE_OUTPUT = 0

if(TABULATE_OUTPUT):

    # Формируем таблицу
    table = []
    for i, name_row in enumerate(filenames):
        row = [name_row] + [f'{distance_matrix[i, j]:.2f}' for j in range(n)]
        table.append(row)
    
    headers = ['']

    # Красивый вывод
    print(tabulate(table, headers, tablefmt="grid"))


CSV_OUTPUT = 0

if(CSV_OUTPUT):

    with open('distance_matrix.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow([''] + filenames)
        for i in range(n):
            w.writerow([filenames[i]] + [f'{distance_matrix[i, j]:.2f}' for j in range(n)])


MIN_DISTANCE_OUT = 1

if(MIN_DISTANCE_OUT):
    
    for i in range(len(filenames)):
        # копируем строку расстояний
        row = distance_matrix[i].copy()
        
        # исключаем расстояние до самого себя
        row[i] = np.inf
        
        # индекс минимального расстояния
        j_min = np.argmin(row)
        
        min_dist = row[j_min]
        
        print(f'{filenames[i]} -> {filenames[j_min]} - {min_dist:.2f}')    
        
        
########################################################################
########################################################################

print(f'Job well done')
