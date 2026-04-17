import cv2
import torch
import numpy as np
import streamlit as st

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

import sqlite3

################################################

MODEL_NAME = 'yolo.v8.nano-face.pt'

DB_PATH = 'faces.sqlite'

DISTANCE_LIMIT = 0.95

################################################

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def align_crop_by_eyes(crop_bgr, left_eye_xy, right_eye_xy):
    lx, ly = left_eye_xy
    rx, ry = right_eye_xy
    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    eyes_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    h, w = crop_bgr.shape[:2]
    return cv2.warpAffine(crop_bgr, M, (w, h), flags=cv2.INTER_LINEAR)

def find_closest(embedding, db_embeddings):
    best_name = None
    best_dist = float('inf')

    for name, db_emb in db_embeddings:
        dist = np.linalg.norm(embedding - db_emb)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name, best_dist

################################################
    
def make_embedding(img):
    
    MIN_FACE_AREA_RATIO = 0.05
    CROP_MARGIN_RATIO   = 0.25
    OUTPUT_SIZE         = (112, 112)

    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w

    detected_results = model_detected(img)

    best = None
    best_area = 0

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
                    left_eye  = (float(kps[0][0]), float(kps[0][1]))
                    right_eye = (float(kps[1][0]), float(kps[1][1]))

                best_area = area
                best = (x1, y1, x2, y2, left_eye, right_eye)

    if best is None:
        return None, 'No face detected'

    if best_area / img_area < MIN_FACE_AREA_RATIO:
        return None, f'Face too small (ratio={best_area / img_area:.4f})'

    x1, y1, x2, y2, left_eye, right_eye = best

    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * CROP_MARGIN_RATIO), int(bh * CROP_MARGIN_RATIO)

    cx1 = clamp(x1 - mx, 0, img_w - 1)
    cy1 = clamp(y1 - my, 0, img_h - 1)
    cx2 = clamp(x2 + mx, 1, img_w)
    cy2 = clamp(y2 + my, 1, img_h)

    crop = img[cy1:cy2, cx1:cx2].copy()

    if left_eye is not None and right_eye is not None:
        leye = (left_eye[0] - cx1, left_eye[1] - cy1)
        reye = (right_eye[0] - cx1, right_eye[1] - cy1)
        ch, cw = crop.shape[:2]
        if 0 <= leye[0] < cw and 0 <= leye[1] < ch \
        and 0 <= reye[0] < cw and 0 <= reye[1] < ch:
            crop = align_crop_by_eyes(crop, leye, reye)

    face = cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).float()
    face_tensor = (face_tensor - 127.5) / 128.0

    with torch.no_grad():
        embedding = model_embedding(face_tensor).cpu().numpy()[0]

    return embedding, 'OK'

################################################

@st.cache_resource
def load_models():
    model_detected  = YOLO(MODEL_NAME)
    model_embedding = InceptionResnetV1(pretrained='vggface2').eval()
    return model_detected, model_embedding

model_detected, model_embedding = load_models()

################################################

@st.cache_resource
def load_embeddings():
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        
        cur.execute("""
            SELECT users.name, embeddings.embedding
            FROM embeddings
            JOIN users ON embeddings.user_id = users.ID
        """) 
               
        rows = cur.fetchall()

    result = []
    for name, blob in rows:
        emb = np.frombuffer(blob, dtype=np.float32)
        result.append((name, emb))

    return result

db_embeddings = load_embeddings()

################################################

if 'last_image_id' not in st.session_state:
    st.session_state.last_image_id = None

col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image('logo.png', width=140)  # меняйте width под нужный размер

with col_title:
    st.markdown(
        '<h1 style="text-align:center; color:#1E90FF;">Identify User</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="border:1px solid red; padding:8px; border-radius:5px; text-align:center;">'
        '<span style="color:red; font-size:12px;">'
        'The system does not store any of your data. Photos are not recorded or saved to disk.'
        '</span></div>',
        unsafe_allow_html=True,
    )    

camera_image = st.camera_input('Camera')

status = st.empty()

if camera_image is not None:
    image_id = hash(camera_image.getvalue())

    if image_id != st.session_state.last_image_id:
        st.session_state.last_image_id = image_id

        status.text('Detecting face...')

        img_bytes = np.frombuffer(camera_image.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        embedding, msg = make_embedding(img_bgr)

        if embedding is None:
            #status.text(f'Failed: {msg}')
            status.markdown('**:red[No face detected]**')
        else:
            name, dist = find_closest(embedding, db_embeddings)
            #status.text(f'Closest: {name} — distance: {dist:.2f}')
            
            if dist > DISTANCE_LIMIT :
                
                status.markdown(f'**:yellow[Unknown user (min.distance: {dist:.2f})]**')
                
            else:    
               
               if( name == 'Cave' ):
                    name = '- Victor -'  # for production - use name instead of nick
               
               status.markdown(f'**:green[User: {name}  (distance: {dist:.2f})]**')
        

print(f'\nOk')
        