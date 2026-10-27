"""Какие модели обслуживает сервис.

Модели задаются переменными окружения - модуль с функцией create():

    PD_MODEL_MODULE=app.models.pd_trained       # обученная модель PD коллеги pd-1.0 (по умолчанию в образе)
    DELAY_MODEL_MODULE=app.models.delay_trained # обученная модель задержки delay-1.0 (по умолчанию в образе)

Переменная не задана, пустая или heuristic - эвристики (heuristic.py): для локального запуска и тестов контракта.

Модель PD:      атрибуты version, trained_at, stub; метод score(buyers, calc_date) -> list[PdResult]
Модель задержки: атрибуты version, trained_at, stub; метод score(request: DelayRequest) -> list[DelayResult]
"""

import importlib
import os
from functools import lru_cache

from .heuristic import HeuristicDelayModel, HeuristicPdModel


def _load(variable: str, default):
    module = (os.getenv(variable) or "").strip()
    if module in ("", "heuristic"):
        return default()
    return importlib.import_module(module).create()


@lru_cache
def pd_model():
    return _load("PD_MODEL_MODULE", HeuristicPdModel)


@lru_cache
def delay_model():
    return _load("DELAY_MODEL_MODULE", HeuristicDelayModel)
