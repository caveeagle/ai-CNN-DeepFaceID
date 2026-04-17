# ====================== STAGE 1: Builder ======================
FROM python:3.11-slim AS builder

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

RUN pip install --no-cache-dir --no-deps facenet-pytorch

# === ultralytics optimisation:
RUN pip uninstall -y matplotlib seaborn pandas 2>/dev/null || true    
    
COPY user_identification.py .
COPY yolo.v8.nano-face.pt .
COPY faces.sqlite .
COPY logo.png .

# Pre-download InceptionResnetV1 weights (~100 MB) at build time
# Makes the image deterministic and avoids downloading on first run
RUN python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2'); print('Weights preloaded!')"

# Cleanup cache to reduce final image size
RUN find /usr/local/lib -name '*.pyc' -delete 2>/dev/null || true && \
    find /usr/local/lib -name '__pycache__' -delete 2>/dev/null || true && \
    rm -rf /root/.cache/pip /root/.cache/torch /root/.cache/huggingface /tmp/* /var/tmp/*

# ====================== STAGE 2: Final (runtime) ======================
FROM python:3.11-slim

WORKDIR /app

# Only required system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy everything needed from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

EXPOSE 8501

CMD ["sh", "-c", "streamlit run user_identification.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
