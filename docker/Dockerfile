FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# === CPU-only PyTorch ===

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY user_identification.py .
COPY yolo.v8.nano-face.pt .
COPY faces.sqlite .
COPY logo.png .

# Pre-download InceptionResnetV1 weights (~100 MB) at build time
# Makes the image deterministic and avoids downloading on first run
RUN python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')"

EXPOSE 8501

CMD ["sh", "-c", "streamlit run user_identification.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
