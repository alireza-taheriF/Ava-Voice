# Ava Voice — container image (foundation).
# syntax=docker/dockerfile:1
#
# Build context is the repository root (see infra/docker-compose.yml:
# `context: ..`). The repository root *is* the ``ava_voice`` package, so the
# source is copied into /app/ava_voice — that way ``ava_voice.main:app``
# resolves with /app on sys.path.

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching. requirements.txt lives at
# the context root, and is placed where the source will land.
COPY requirements.txt ./ava_voice/requirements.txt
RUN pip install --upgrade pip && pip install -r ava_voice/requirements.txt

# Copy the application source into the importable package directory.
COPY . ./ava_voice/

# Run as an unprivileged user; give it ownership of the log directory only.
RUN useradd --create-home --shell /usr/sbin/nologin ava \
    && mkdir -p /app/ava_voice/logs \
    && chown -R ava:ava /app/ava_voice/logs
USER ava

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

# Run the ASGI app. Override CMD for workers/reload as needed.
CMD ["uvicorn", "ava_voice.main:app", "--host", "0.0.0.0", "--port", "8000"]
