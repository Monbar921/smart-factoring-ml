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

## LLM (GigaChat) — только через backend

Сервис моделей не ходит в GigaChat напрямую: ключ, токен, лимит одновременных запросов и выбор модели — в backend.
Если модели нужна LLM, вызывайте её через [service/app/llm.py](service/app/llm.py):

```python
from app.llm import backend_llm, LlmUnavailable

try:
    text = backend_llm().chat([{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
                              temperature=0.2, max_tokens=300).content  # model="GigaChat-2-Pro" - необязательно
except LlmUnavailable:
    text = None  # LLM выключена в backend (LLM_MODE не gigachat) или недоступна — работаем без неё
```

Адрес — `BACKEND_LLM_URL`, по умолчанию `http://backend:8080/internal/rest/llm` (сеть стенда). Контракт —
`docs/claude/module-0/backend-api.md` в smart-factoring-backend, раздел «LLM для сервиса моделей».

## Запуск

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
cd service && ../.venv/Scripts/python -m pytest -q
../.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

Docker: `docker build -t smart-factoring-ml .` → порт 8000, healthcheck `/v1/health`, процесс не от root.

## Эксплуатация

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `WEB_CONCURRENCY` | `1` | процессов uvicorn (оценки считаются на CPU — не больше выделенных ядер) |
| `LOG_LEVEL` | `INFO` | `DEBUG` — ещё и строка на каждую пачку |
| `ML_MAX_BATCH` | `2000` | элементов в пачке, больше — `413 BATCH_TOO_LARGE` (backend шлёт по 200) |
| `PD_MODEL_MODULE`, `DELAY_MODEL_MODULE` | пусто | обученные модели, см. выше; пусто — эвристики |
| `BACKEND_LLM_URL` | `http://backend:8080/internal/rest/llm` | LLM через backend |

- Модели загружаются при старте: если модуль из `PD_MODEL_MODULE` / `DELAY_MODEL_MODULE` не загрузился, процесс
  завершается с ошибкой (на стенде контейнер не станет healthy, deploy упадёт), а не отвечает молча эвристиками.
- `/v1/health` — `{"status": "UP", "pd": "<версия>", "delay": "<версия>"}`.
- Лог: строка на запрос с длительностью и `X-Request-ID` (из запроса или новый, возвращается в ответе).
- Объём демо (100 клиентов, 1 500 открытых УПД): шаг оценки прогона backend < 1 с, пачка 200 УПД — до 70 мс.

## CI/CD и деплой

[.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml): PR в `main`/`test` — тесты; push — ещё образ
`ghcr.io/monbar921/smart-factoring-ml:<short sha>` (и тег ветки), smoke-проверка `/v1/health` и `repository_dispatch`
в smart-factoring-deploy, как у остальных сервисов. В репозитории нужны `vars.DEPLOY_REPO` и `secrets.DEPLOY_TOKEN`.
На стенде сервис `ml` без внешних портов, backend ходит в него по `http://ml:8000` (`ML_MODE=http`).
