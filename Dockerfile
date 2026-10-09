FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    BLOODSMEAR_MODEL_DIR=/app/models/blood-cell-yolo11/1.0.0 \
    BLOODSMEAR_OUTPUT_DIR=/app/outputs \
    BLOODSMEAR_HOST=0.0.0.0 \
    BLOODSMEAR_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --create-home app

WORKDIR /app

COPY requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir -r /app/requirements.lock

COPY src /app/src
RUN mkdir -p /app/outputs && chown -R app:app /app/outputs

USER app

EXPOSE 8000

CMD ["python", "-m", "bloodsmear.cli", "serve"]

