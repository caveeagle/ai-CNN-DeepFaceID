# DeepFaceID

DeepFaceID is a face recognition system that identifies users through a laptop camera.

![DeepFaceID](logo.png)

## How It Works

The system uses two algorithms:
- **YOLOv8-face** — detects faces and returns keypoints (eye positions) for alignment
- **InceptionResnetV1** (pretrained on VGGFace2) — extracts a 512-dimensional embedding vector from each face crop

Each detected face is cropped, aligned by eye position, resized to 112×112, and passed to the embedding model. Identification is done by computing the L2 distance between the query embedding and all stored embeddings. If the closest distance is below a calibrated threshold, the user is recognized; otherwise the person is treated as unknown.

Embeddings are stored in a lightweight SQLite database. User photos are not stored — they are deleted after embedding extraction.

## Workflow

The system consists of three main scripts used in sequence:

1. **`camera.py`** — a desktop GUI app for capturing several photos of a user via webcam. Photos are saved to the `images/` directory with a name-based filename (e.g. `John-001.png`, `John-002.png`)

2. **`add_users.py`** — a batch script that reads all photos from `images/`, extracts face embeddings, and saves user names and embeddings to the database. Already processed files are skipped automatically.

3. **`user_identification.py`** — a Streamlit web app for real-time identification. The user takes a snapshot via webcam; the system computes the embedding and returns the closest match from the database.

## Test Scripts

Scripts with the `test_` prefix were used during development for parameter tuning, library verification, and proof-of-concept work. They are not required for normal system operation.

## Project Files

| File | Description |
|------|-------------|
| `camera.py` | Desktop GUI app (tkinter) for capturing photos from a webcam. Saves snapshots to the `images/` directory with a name-based filename pattern (e.g. `John-001.png`) |
| `add_users.py` | Batch script that processes all images in the `images/` directory: extracts face embeddings and saves them to the database, skipping files already processed |
| `user_identification.py` | Streamlit web app for real-time face identification via webcam. Captures a snapshot, computes the embedding, and finds the closest match in the database |
| `test_yolo.py` | Basic test of the YOLO face detection model on a single image |
| `test_cycle_detection.py` | Iterates over all images in the `images/` directory and runs face detection, displaying results one by one. For testing purposes only |
| `test_cycle_embedding.py` | Runs face detection and embedding extraction on all images in `images/`, with optional saving to the database |
| `test_distance.py` | Computes L2 distances between embeddings stored in the database. Used for threshold calibration |
| `test_distance_grid.py` | Extended version of `test_distance.py` — computes a full distance matrix across multiple embeddings |
| `test_lib.py` | Sanity check for installed libraries (OpenCV, PyTorch, ultralytics, facenet-pytorch) |
| `yolo.v8.nano-face.pt` | Pretrained YOLOv8 nano model fine-tuned for face detection with keypoints |
| `faces.sqlite` | SQLite database storing users, face embeddings and source filenames |
| `requirements.txt` | Python dependencies |
