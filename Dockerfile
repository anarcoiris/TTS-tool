# Dockerfile
# GPU / CUDA 11.8 + Python 3.10 + Coqui TTS
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-setuptools python3-dev \
    build-essential git ffmpeg libsndfile1 wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# asegurar que "python" apunta a python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# pip actualizada
RUN python -m pip install --upgrade pip setuptools wheel

# instalar PyTorch con soporte CUDA 11.8 (usa index-url oficial)
RUN python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# instalar Coqui TTS y utilidades de audio
# Si quieres limitar versiones o instalar desde repo, cámbialo aquí.
RUN python -m pip install --upgrade TTS soundfile librosa numpy matplotlib

# Crear workspace y copiar (opcional) tu código
WORKDIR /workspace
# copia todo (puedes excluir con .dockerignore)
COPY . /workspace

# Crear un usuario no-root opcional (mejor práctica)
RUN useradd -ms /bin/bash ttsuser && chown -R ttsuser:ttsuser /workspace
USER ttsuser

# Entrypoint por defecto: abrir bash (para que puedas ejecutar comandos interactivos)
ENTRYPOINT ["/bin/bash"]

RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
