FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install only minimal production dependencies to save RAM and disk space
# Force CPU-only versions of PyTorch and Torchvision
RUN pip install --no-cache-dir \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    python-multipart>=0.0.6 \
    opencv-python-headless>=4.8.0 \
    Pillow>=10.0.0 \
    numpy>=1.24.0 \
    pyyaml>=6.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch>=2.0.0 \
    torchvision>=0.15.0

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
