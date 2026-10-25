# smart-factoring-ml

Сервис моделей Smart Factoring: PD покупателя и задержка оплаты по УПД (блок 4 BRD).

- Контракт: [service/contract/ml-service-openapi.yaml](service/contract/ml-service-openapi.yaml) (источник — smart-factoring-backend `docs/claude/module-0`).
- Сервис stateless: backend присылает **сырые данные** пачками (отчётность, история УПД), признаки считаются здесь
  ([service/app/features.py](service/app/features.py)) — тем же кодом, что при обучении.
- Пока нет обученных моделей, работают эвристики ([service/app/models/heuristic.py](service/app/models/heuristic.py)),
  в `/v1/models` они помечены `stub: true`.

## Как подключить обученные модели

1. Положить модуль, например `service/app/models/pd_trained.py`, с функцией `create()`, которая возвращает объект:
   - атрибуты `version`, `trained_at`, `stub = False`;
   - PD: `score(buyers: list[Buyer], calc_date) -> list[PdResult]` — базовый PD и «мало данных» по п. 4.2
     (готово: `features.low_data_reason`), три причины с направлением (SHAP);
   - задержка: `score(request: DelayRequest) -> list[DelayResult]` — `NO_PAYMENT_HISTORY`, если у покупателя нет ни одного
     оплаченного УПД; `no_pair_history`, если нет истории у пары (п. 4.5).
2. Задать переменные `PD_MODEL_MODULE=app.models.pd_trained`, `DELAY_MODEL_MODULE=app.models.delay_trained`.
3. В backend включить `ML_MODE=http`, `ML_BASE_URL=http://smart-factoring-ml:8000`.

## Запуск

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
cd service && ../.venv/Scripts/python -m pytest -q
../.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

Docker: `docker build -t smart-factoring-ml .` → порт 8000, healthcheck `/v1/health`.
