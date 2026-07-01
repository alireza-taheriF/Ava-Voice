# Ava Voice — container image (foundation).
# syntax=docker/dockerfile:1

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY ava_voice/requirements.txt ./ava_voice/requirements.txt
RUN pip install --upgrade pip && pip install -r ava_voice/requirements.txt

# Copy the application source.
COPY . .

EXPOSE 8000

# Run the ASGI app. Override CMD for workers/reload as needed.
CMD ["uvicorn", "ava_voice.main:app", "--host", "0.0.0.0", "--port", "8000"]
