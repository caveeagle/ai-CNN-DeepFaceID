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
