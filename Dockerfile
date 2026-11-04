FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    OUTPUT_DIR=/tmp/outputs \
    OUTPUT_TTL_HOURS=24 \
    DAILY_LIMIT_PER_VISITOR=10 \
    DAILY_LIMIT_TOTAL=80 \
    SECURE_COOKIES=1

# libglib is the one system library OpenCV (headless) still needs on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY static ./static
COPY examples/sample_input.png examples/line.png ./examples/

# Run as an unprivileged user.
RUN useradd --create-home studio
USER studio

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"

# --proxy-headers so client IPs and https are seen correctly behind the host's proxy.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
