"""Настройки сервиса из переменных окружения (docker-compose стенда, см. README)."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # уровень логов: DEBUG, INFO, WARNING
    log_level: str
    # предел элементов в одной пачке: backend шлёт по smart-factoring.ml.batch-size (200), больше - 413
    max_batch: int


def load() -> Settings:
    return Settings(
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        max_batch=int(os.getenv("ML_MAX_BATCH", "2000")),
    )


settings = load()
