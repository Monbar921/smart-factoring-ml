"""Сервис моделей Smart Factoring: PD покупателя и задержка оплаты по УПД (блок 4 BRD).

Контракт - contract/ml-service-openapi.yaml. Сервис stateless (Р-20): backend присылает сырые данные пачками.
Ошибка по отдельному элементу не роняет пачку: элемент получает status = ERROR.
Модели загружаются при старте: если PD_MODEL_MODULE / DELAY_MODEL_MODULE не загрузились, сервис не стартует.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .models.registry import delay_model, pd_model
from .schemas import DelayRequest, DelayResponse, DelayResult, ModelsInfo, ModelVersion, PdRequest, PdResponse, PdResult
from .settings import settings

SUPPORTED_THRESHOLDS = [7, 14, 30]

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("smart-factoring-ml")


@asynccontextmanager
async def lifespan(_: FastAPI):
    pd, delay = pd_model(), delay_model()
    log.info("Модели загружены: PD %s (stub=%s), задержка %s (stub=%s); пачка до %d",
             pd.version, pd.stub, delay.version, delay.stub, settings.max_batch)
    yield


app = FastAPI(title="Smart Factoring — сервис моделей", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Строка лога на запрос (кроме healthcheck) и X-Request-ID: из запроса или новый."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    if request.url.path != "/v1/health":
        log.info("%s %s %d %d мс rid=%s", request.method, request.url.path, response.status_code,
                 round((time.perf_counter() - started) * 1000), request_id)
    return response


@app.get("/v1/health")
def health() -> dict:
    pd, delay = pd_model(), delay_model()
    return {"status": "UP", "pd": pd.version, "delay": delay.version}


@app.get("/v1/models")
def models() -> ModelsInfo:
    pd, delay = pd_model(), delay_model()
    return ModelsInfo(
        pd=ModelVersion(version=pd.version, trained_at=pd.trained_at, stub=pd.stub),
        delay=ModelVersion(version=delay.version, trained_at=delay.trained_at, stub=delay.stub),
        supported_delay_thresholds=SUPPORTED_THRESHOLDS,
    )


@app.post("/v1/pd/score")
def score_pd(request: PdRequest) -> PdResponse:
    check_batch(len(request.buyers))
    model = pd_model()
    results = []
    for buyer in request.buyers:
        try:
            results.extend(model.score([buyer], request.calc_date))
        except Exception as error:  # noqa: BLE001 - ошибка одного покупателя не роняет пачку
            log.exception("PD %s", buyer.inn)
            results.append(PdResult(inn=buyer.inn, status="ERROR", error=str(error)))
    log_batch("PD", results)
    return PdResponse(model_version=model.version, results=results)


@app.post("/v1/delay/score")
def score_delay(request: DelayRequest) -> DelayResponse:
    if request.delay_threshold_days not in SUPPORTED_THRESHOLDS:
        raise HTTPException(status_code=400, detail={"code": "UNSUPPORTED_THRESHOLD", "message": "порог 7, 14 или 30"})
    check_batch(len(request.upds))
    model = delay_model()
    results = []
    for upd in request.upds:
        try:
            results.extend(model.score(request.model_copy(update={"upds": [upd]})))
        except Exception as error:  # noqa: BLE001
            log.exception("Задержка %s", upd.upd_id)
            results.append(DelayResult(upd_id=upd.upd_id, status="ERROR", error=str(error)))
    log_batch("Задержка", results)
    return DelayResponse(model_version=model.version, results=results)


def check_batch(size: int) -> None:
    if size > settings.max_batch:
        raise HTTPException(status_code=413, detail={
            "code": "BATCH_TOO_LARGE", "message": f"в пачке {size} элементов, допустимо до {settings.max_batch}"})


def log_batch(what: str, results: list) -> None:
    errors = sum(1 for result in results if result.status == "ERROR")
    log.log(logging.WARNING if errors else logging.DEBUG, "%s: %d элементов, ошибок %d", what, len(results), errors)
