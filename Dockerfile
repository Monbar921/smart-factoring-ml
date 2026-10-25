# Сервис моделей Smart Factoring (контракт - service/contract/ml-service-openapi.yaml)
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/app ./app

# обученные модели: PD_MODEL_MODULE / DELAY_MODEL_MODULE (см. service/app/models/registry.py)
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
