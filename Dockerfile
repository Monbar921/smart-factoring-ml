# Сервис моделей Smart Factoring (контракт - service/contract/ml-service-openapi.yaml)
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 - OpenMP для LightGBM (в slim-образе его нет)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY service/app ./app

# не root: сервису не нужна запись на диск
RUN useradd --system --uid 10001 --no-create-home ml
USER ml

# обученные модели коллеги (pd-1.0, delay-1.0) - по умолчанию; пустые значения - эвристики (service/app/models/registry.py)
ENV PD_MODEL_MODULE=app.models.pd_trained     DELAY_MODEL_MODULE=app.models.delay_trained
# WEB_CONCURRENCY - число процессов uvicorn (по умолчанию 1), LOG_LEVEL, ML_MAX_BATCH - см. service/app/settings.py
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health')"
# access log uvicorn выключен: строку на запрос пишет сам сервис (с длительностью и X-Request-ID)
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-1} --no-access-log"]
