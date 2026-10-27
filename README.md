# smart-factoring-ml

Сервис моделей Smart Factoring: PD покупателя и задержка оплаты по УПД (блок 4 BRD).

- Контракт: [service/contract/ml-service-openapi.yaml](service/contract/ml-service-openapi.yaml) (источник — smart-factoring-backend `docs/claude/module-0`).
- Сервис stateless: backend присылает **сырые данные** пачками (отчётность, история УПД), признаки считаются здесь
  ([service/app/features.py](service/app/features.py)) — тем же кодом, что при обучении.
- Пока нет обученных моделей, работают эвристики ([service/app/models/heuristic.py](service/app/models/heuristic.py)),
  в `/v1/models` они помечены `stub: true`.

## Модели

В образе по умолчанию — обученные модели коллеги (ветка `feature/SMF-34`, перенесены в `service/app/models`):

| | PD покупателя `pd-1.0` | Задержка оплаты `delay-1.0` |
|---|---|---|
| Модуль | `app.models.pd_trained` (`PD_MODEL_MODULE`) | `app.models.delay_trained` (`DELAY_MODEL_MODULE`) |
| Файлы модели | `app/models/pd/`: `pd_model.pkl`, `pd_features.py`, `pd_model.py`, README | `app/models/delay/`: `delay_model_1_0.pkl`, `dl_common.py`, `dl_model.py`, `alloc.py`, README |
| Что внутри | LightGBM + калибровка Платта, три причины по SHAP | 4 бустинга (p_7, p_14, p_30, p_90) + регрессия ожидаемых дней |

Значение `heuristic` (или пустое, как в тестах контракта) — эвристики `heuristic.py`.

**PD.** Признаки строятся из отчётности в запросе (строки форм за год отчётности T = год до даты расчёта и два
предыдущих, в тыс. руб., как при обучении на RFSD), а не по ИНН из хранилища, как `score_inn.py` коллеги.
Где модель не определена, — базовый PD эвристики по разделу ОКВЭД с пометкой «мало данных»: нет отчётности за прошлый
год, выручка ниже 120 млн, компания моложе года, разделы ОКВЭД K и O (`EXCLUDED_INDUSTRY`, их не было в обучении).
Регион переводится в вид RFSD (`moscow city`, `moscow reg.`, `sankt-petersburg`): модель различает только их.

**Задержка.** Переходник JSON → таблицы модели (README модели, «Подключение к сервису»): УПД и даты их закрытия — из
`history`, как их прислал backend (сверка оплат заново не делается); оплаченная часть = `amount − outstanding`;
уровень A–D покупателя для признака — по порогам модели 0,4 % / 1,5 % / 6 % (в backend для решения свои пороги);
коэффициенты покупателя — по последней отчётности из `buyers`; отрасль клиента — `client_industry` (коды backend
переводятся в значения обучения). Закрытый или просроченный УПД — `ERROR`: модель для него не определена.

**Сверка с эталоном.** На тестовом наборе коллеги (`feature/SMF-35`, партии 01–05, 5 936 УПД на 30.12.2025) переходник
даёт те же статусы и те же числа, что `delay_scores.csv`: |Δp| ≤ 5·10⁻⁷, |Δ expected_days| ≤ 0,005 — точность самого
эталона. В тестах — подмножество партии 01 (`service/tests/data/delay_batch01`) и через ядро, и через JSON API.

Новая версия модели — заменить файлы в `app/models/pd` / `app/models/delay` (имена модулей в pickle сопоставляются
в `app/models/_pickle.py`) и прогнать тесты; другой модуль — свой `create()` с `version`, `trained_at`, `stub = False`
и `score(...)` как в `registry.py`.

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
# Python 3.13, как в образе: под него собраны колёса lightgbm / numpy / pandas, на которых обучали модели
uv venv --python 3.13 .venv313
uv pip install --python .venv313/Scripts/python.exe -r requirements-dev.txt
cd service && ../.venv313/Scripts/python -m pytest -q
../.venv313/Scripts/python -m uvicorn app.main:app --port 8000
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
