import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

################################################

MIN_FACE_AREA_RATIO = 0.05
CROP_MARGIN_RATIO   = 0.25
OUTPUT_SIZE         = (112, 112)

MODEL_DETECTED_PATH = 'yolo.v8.nano-face.pt'

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

def make_embedding(img):
    """
    Accept a BGR numpy image, return (embedding, msg).
    """
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w

    detected_results = st.session_state.model_detected(img)

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
        embedding = st.session_state.model_embedding(face_tensor).cpu().numpy()[0]

    return embedding, 'OK'

################################################

# Load models once into session_state
if 'model_detected' not in st.session_state:
    st.session_state.model_detected  = YOLO(MODEL_DETECTED_PATH)
    st.session_state.model_embedding = InceptionResnetV1(pretrained='vggface2').eval()

################################################

st.title('Identify User')

camera_image = st.camera_input('Camera')

col1, col2 = st.columns([1, 3])

with col1:
    identify = st.button('Identify User')

with col2:
    status = st.empty()

if identify:
    if camera_image is None:
        status.text('No image from camera')
    else:
        status.text('Detecting face...')

        img_bytes = np.frombuffer(camera_image.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        embedding, msg = make_embedding(img_bgr)

        if embedding is None:
            status.text(f'Failed: {msg}')
        else:
            status.text(f'Embedding ready — {len(embedding)} dims')

        # TODO: compare embedding with database
        
        