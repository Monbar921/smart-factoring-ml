"""Какие модели обслуживает сервис.

По умолчанию - эвристики (heuristic.py). Обученные модели подключаются переменными окружения:

    PD_MODEL_MODULE=app.models.pd_trained       # модуль с функцией create() -> модель PD
    DELAY_MODEL_MODULE=app.models.delay_trained # модуль с функцией create() -> модель задержки

Модель PD:      атрибуты version, trained_at, stub; метод score(buyers, calc_date) -> list[PdResult]
Модель задержки: атрибуты version, trained_at, stub; метод score(request: DelayRequest) -> list[DelayResult]
"""

import importlib
import os
from functools import lru_cache

from .heuristic import HeuristicDelayModel, HeuristicPdModel


def _load(variable: str, default):
    module = os.getenv(variable)
    if not module:
        return default()
    return importlib.import_module(module).create()


@lru_cache
def pd_model():
    return _load("PD_MODEL_MODULE", HeuristicPdModel)


@lru_cache
def delay_model():
    return _load("DELAY_MODEL_MODULE", HeuristicDelayModel)
