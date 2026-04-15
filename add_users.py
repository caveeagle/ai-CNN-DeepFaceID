import os
import sqlite3

import cv2
from ultralytics import YOLO

import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1

################################################
# Models
################################################

file_model_detected = 'yolo.v8.nano-face.pt'

model_detected = YOLO(file_model_detected)
model_embedding = InceptionResnetV1(pretrained='vggface2').eval()

################################################
# Helpers
################################################

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def align_crop_by_eyes(crop_bgr, left_eye_xy, right_eye_xy):
    lx, ly = left_eye_xy
    rx, ry = right_eye_xy

    dx = rx - lx
    dy = ry - ly
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    h, w = crop_bgr.shape[:2]
    return cv2.warpAffine(crop_bgr, M, (w, h), flags=cv2.INTER_LINEAR)

def extract_username(filename):
    """
    Extract user name from filename.
    Everything before the LAST dash is considered the user name.
    Example:
      Aleksei-007.png -> Aleksei
      Jean-Claude-Van-001.jpg -> Jean-Claude-Van
    """
    name = os.path.splitext(filename)[0]
    if '-' not in name:
        return name
    return name.rsplit('-', 1)[0]

def ensure_user(name: str) -> int:
    """
    Return user ID by name.
    If user does not exist - insert, update cache, and return new ID.
    """
    if name in USER_CACHE:
        return USER_CACHE[name]
    
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (name) VALUES (?)", (name,))
        conn.commit()
        uid = cur.lastrowid

    USER_CACHE[name] = uid
    return uid

################################################
# Embeddings
################################################

def make_embedding(filename):
    
    print(f'Processed: {filename}')
    
    path = os.path.join('images', filename)
    img = cv2.imread(path)

    if img is None:
        return None

    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w

    ################################################

    detected_results = model_detected(img)

    best = None
    best_area = 0

    MIN_FACE_AREA_RATIO = 0.05
    CROP_MARGIN_RATIO = 0.25
    OUTPUT_SIZE = (112, 112)

    for r in detected_results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        has_kps = (getattr(r, 'keypoints', None) is not None) and (r.keypoints.xy is not None)

        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                left_eye = right_eye = None

                if has_kps and i < len(r.keypoints.xy):
                    kps = r.keypoints.xy[i]
                    # expected: kps[0]=left_eye, kps[1]=right_eye
                    left_eye  = (float(kps[0][0]), float(kps[0][1]))
                    right_eye = (float(kps[1][0]), float(kps[1][1]))

                best_area = area
                best = (x1, y1, x2, y2, left_eye, right_eye)

    if best is None:
        print(f'{filename}: skipped — no faces found')
        return None

    if best_area / img_area < MIN_FACE_AREA_RATIO:
        print(f'{filename}: skipped — face too small (ratio={best_area / img_area:.4f})')
        return None

    x1, y1, x2, y2, left_eye, right_eye = best

    ################################################

    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * CROP_MARGIN_RATIO)
    my = int(bh * CROP_MARGIN_RATIO)

    cx1 = clamp(x1 - mx, 0, img_w - 1)
    cy1 = clamp(y1 - my, 0, img_h - 1)
    cx2 = clamp(x2 + mx, 1, img_w)
    cy2 = clamp(y2 + my, 1, img_h)

    crop = img[cy1:cy2, cx1:cx2].copy()

    # Align by eyes if keypoints are available
    if left_eye is not None and right_eye is not None:
        leye = (left_eye[0] - cx1, left_eye[1] - cy1)
        reye = (right_eye[0] - cx1, right_eye[1] - cy1)

        ch, cw = crop.shape[:2]
        if 0 <= leye[0] < cw and 0 <= leye[1] < ch \
        and 0 <= reye[0] < cw and 0 <= reye[1] < ch:
            crop = align_crop_by_eyes(crop, leye, reye)

    ################################################

    face = cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).float()

    # Normalize to [-1, 1] as expected by FaceNet 
    face_tensor = (face_tensor - 127.5) / 128.0

    with torch.no_grad():
        embedding = model_embedding(face_tensor).cpu().numpy()[0]

    return embedding
    
################################################
# Load images
################################################

IMAGE_DIR = 'images'

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

if not image_files:
    raise RuntimeError('No images found')

################################################
# Database
################################################

DB_PATH = 'faces.sqlite'

with sqlite3.connect(DB_PATH) as conn:
    cur = conn.cursor()

    cur.execute("SELECT ID, name FROM users")
    USER_CACHE = {name: uid for uid, name in cur.fetchall()}

    cur.execute("SELECT filename FROM embeddings")
    PROCESSED_FILES = {row[0] for row in cur.fetchall()}
        
################################################
# Processing loop
################################################

records = []

for filename in image_files:

    if filename in PROCESSED_FILES:
        print(f'{filename}: skipped — already in DB')
        continue
            
    user_name = extract_username(filename)
    
    uid = ensure_user(user_name)
    
    embedding = make_embedding(filename)
    
    if embedding is None:
        print(f'{filename}: skipped — no embedding')
        continue
    
    records.append((uid, embedding.astype(np.float32).tobytes(), filename))

SAVE_TO_DB = 1

if(SAVE_TO_DB):
      
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT INTO embeddings (user_id, embedding, filename) VALUES (?, ?, ?)",
            records
        )        
        conn.commit()

########################################

print('\nJob finished successfully')
