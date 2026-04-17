# ====================== STAGE 1: Builder (всё собираем здесь) ======================
FROM python:3.11-slim AS builder

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# === CPU-only PyTorch ===
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Установка зависимостей из requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# facenet-pytorch отдельно (как у тебя было)
RUN pip install --no-cache-dir --no-deps facenet-pytorch

# Копируем файлы приложения
COPY user_identification.py .
COPY yolo.v8.nano-face.pt .
COPY faces.sqlite .
COPY logo.png .

# Предзагрузка весов FaceNet
RUN python -c "
from facenet_pytorch import InceptionResnetV1
_ = InceptionResnetV1(pretrained='vggface2')
print('FaceNet weights preloaded successfully')
"

# ====================== STAGE 2: Final (минимальный runtime) ======================
FROM python:3.11-slim

WORKDIR /app

# Только необходимые системные библиотеки
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем всё, что нужно, из builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

EXPOSE 8501

CMD ["sh", "-c", "streamlit run user_identification.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
