"""Сервис моделей Smart Factoring: PD покупателя и задержка оплаты по УПД (блок 4 BRD).

Контракт - contract/ml-service-openapi.yaml. Сервис stateless (Р-20): backend присылает сырые данные пачками.
Ошибка по отдельному элементу не роняет пачку: элемент получает status = ERROR.
"""

import logging

from fastapi import FastAPI, HTTPException

from .models.registry import delay_model, pd_model
from .schemas import DelayRequest, DelayResponse, DelayResult, ModelsInfo, ModelVersion, PdRequest, PdResponse, PdResult

SUPPORTED_THRESHOLDS = [7, 14, 30]

log = logging.getLogger("smart-factoring-ml")
app = FastAPI(title="Smart Factoring — сервис моделей", version="1.0")


@app.get("/v1/health")
def health() -> dict:
    pd_model()
    delay_model()
    return {"status": "UP"}


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
    model = pd_model()
    results = []
    for buyer in request.buyers:
        try:
            results.extend(model.score([buyer], request.calc_date))
        except Exception as error:  # noqa: BLE001 - ошибка одного покупателя не роняет пачку
            log.exception("PD %s", buyer.inn)
            results.append(PdResult(inn=buyer.inn, status="ERROR", error=str(error)))
    return PdResponse(model_version=model.version, results=results)


@app.post("/v1/delay/score")
def score_delay(request: DelayRequest) -> DelayResponse:
    if request.delay_threshold_days not in SUPPORTED_THRESHOLDS:
        raise HTTPException(status_code=400, detail={"code": "UNSUPPORTED_THRESHOLD", "message": "порог 7, 14 или 30"})
    model = delay_model()
    results = []
    for upd in request.upds:
        try:
            results.extend(model.score(request.model_copy(update={"upds": [upd]})))
        except Exception as error:  # noqa: BLE001
            log.exception("Задержка %s", upd.upd_id)
            results.append(DelayResult(upd_id=upd.upd_id, status="ERROR", error=str(error)))
    return DelayResponse(model_version=model.version, results=results)
