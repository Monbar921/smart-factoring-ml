# Сервис моделей Smart Factoring (контракт - service/contract/ml-service-openapi.yaml)
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY service/app ./app

# не root: сервису не нужна запись на диск
RUN useradd --system --uid 10001 --no-create-home ml
USER ml

# обученные модели: PD_MODEL_MODULE / DELAY_MODEL_MODULE (см. service/app/models/registry.py)
# WEB_CONCURRENCY - число процессов uvicorn (по умолчанию 1), LOG_LEVEL, ML_MAX_BATCH - см. service/app/settings.py
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health')"
# access log uvicorn выключен: строку на запрос пишет сам сервис (с длительностью и X-Request-ID)
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-1} --no-access-log"]
