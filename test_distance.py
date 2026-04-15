import sqlite3
import numpy as np

db = 'faces.sqlite'

with sqlite3.connect(db) as conn:
    cur = conn.cursor()

    # 1. Получить embedding с active=2 (ожидается только один)
    cur.execute("SELECT filename, embedding FROM testing WHERE active = 2 LIMIT 1")
    row = cur.fetchone()
    if row is None:
        raise Exception("Нет записей с active=2!")
    fname2, emb_blob2 = row
    emb2 = np.frombuffer(emb_blob2, dtype=np.float32)

    # 2. Получить все embedding'и с active=1
    cur.execute("SELECT filename, embedding FROM testing WHERE active = 1")
    rows1 = cur.fetchall()
    embeddings_1 = [(fname, np.frombuffer(emb_blob, dtype=np.float32)) for fname, emb_blob in rows1]

# 3. Считаем расстояния
print(f"\nСравнения для: {fname2}")
for fname1, emb1 in embeddings_1:
    l2_dist = np.linalg.norm(emb2 - emb1)
    print(f"{fname1} - L2: {l2_dist:.4f}")
