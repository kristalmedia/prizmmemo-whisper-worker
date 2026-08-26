FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    WHISPER_MODEL_ID=Systran/faster-whisper-large-v3 \
    WHISPER_COMPUTE_TYPE=float16 \
    WHISPER_BATCH_SIZE=16 \
    MAX_AUDIO_BYTES=536870912 \
    ALLOWED_AUDIO_HOST_SUFFIX=.r2.cloudflarestorage.com

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
