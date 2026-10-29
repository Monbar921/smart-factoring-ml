"""Обученные модели коллеги (pd-1.0, delay-1.0) за контрактом сервиса.

Эталон задержки - delay_scores.csv тестового набора коллеги (feature/SMF-35, ml/delay/delivery/batch_01):
в tests/data/delay_batch01 - все УПД и платежи 8 покупателей партии и их эталонные оценки на 30.12.2025.
Совпадение - до точности эталона: p_* - 6 знаков, expected_days - 2.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.models import delay_trained, pd_trained, registry
from app.models.delay import dl_common as dc

DATA = Path(__file__).parent / "data" / "delay_batch01"
CALC_DATE = "2025-12-30"
P_TOLERANCE = 5e-7 + 1e-9
DAYS_TOLERANCE = 5e-3 + 1e-9


@pytest.fixture(scope="module")
def delay():
    return delay_trained.create()


@pytest.fixture(scope="module")
def batch():
    tab = dc.load_csv_batch(str(DATA))
    reference = pd.read_csv(DATA / "delay_scores.csv", dtype={"upd_id": str})
    t = int(dc.day_int(pd.Series([CALC_DATE]))[0])
    upds, allocations = dc.reconcile(tab, upto=t)
    return tab, reference, t, upds, allocations


@pytest.fixture
def trained(monkeypatch):
    monkeypatch.setenv("PD_MODEL_MODULE", "app.models.pd_trained")
    monkeypatch.setenv("DELAY_MODEL_MODULE", "app.models.delay_trained")
    registry.pd_model.cache_clear()
    registry.delay_model.cache_clear()
    yield TestClient(main.app)
    registry.pd_model.cache_clear()
    registry.delay_model.cache_clear()


def paid_on(upds, allocations, t):
    paid = allocations[allocations.d <= t].groupby("u").a.sum()
    return {upd_id: float(paid.get(i, 0)) / 100 for i, upd_id in enumerate(upds.upd_id)}


def test_delay_core_matches_reference(delay, batch):
    tab, reference, t, upds, allocations = batch
    paid = paid_on(upds, allocations, t)
    columns = ["upd_id", "upd_number", "client_id", "buyer_inn", "upd_d", "plan_d", "amount_k", "close_d"]

    out = delay_trained.score_frame(delay.model, upds[columns], tab["cl"], tab["by"],
                                    {upd_id: paid[upd_id] for upd_id in reference.upd_id}, t)

    merged = reference.set_index("upd_id").join(out, rsuffix="_our")
    assert (merged.status == merged.status_our).all()
    scored = merged[merged.status == "ok"]
    for column in ("p_7", "p_14", "p_30", "p_90"):
        assert np.abs(scored[column] - scored[column + "_our"].astype(float)).max() <= P_TOLERANCE, column
    assert np.abs(scored.expected_days - scored.expected_days_our.astype(float)).max() <= DAYS_TOLERANCE


def test_delay_api_matches_reference(trained, batch):
    """Полный путь: JSON запрос, как его собирает backend (история с датами закрытия, отчётность покупателя)."""
    tab, reference, t, upds, allocations = batch
    paid = paid_on(upds, allocations, t)
    day = lambda value: str(dc.int_day(value).date())  # noqa: E731
    industry = dict(zip(tab["cl"].client_id, tab["cl"].industry))
    by = tab["by"].set_index("buyer_inn")
    report_year = pd.read_csv(DATA / "buyers.csv", dtype=str).set_index("buyer_inn").report_year.astype(int)
    scored = set(reference.upd_id)
    history = [{
        "upd_id": row.upd_id, "number": row.upd_number, "client_inn": row.client_id, "buyer_inn": row.buyer_inn,
        "amount": row.amount_k / 100, "upd_date": day(row.upd_d), "planned_due_date": day(row.plan_d),
        "status": "OPEN" if np.isnan(row.close_d) else "PAID",
        "paid_date": None if np.isnan(row.close_d) else day(row.close_d),
        "outstanding": row.amount_k / 100 - paid[row.upd_id] if np.isnan(row.close_d) else 0,
    } for row in upds.itertuples()]
    request_upds = [{
        "upd_id": row.upd_id, "number": row.upd_number, "client_inn": row.client_id, "client_industry": industry[row.client_id],
        "buyer_inn": row.buyer_inn, "buyer_pd": float(by.loc[row.buyer_inn, "pd"]), "amount": row.amount_k / 100,
        "outstanding": row.amount_k / 100 - paid[row.upd_id], "upd_date": day(row.upd_d), "planned_due_date": day(row.plan_d),
        "deferral_days": int(row.plan_d - row.upd_d),
    } for row in upds.itertuples() if row.upd_id in scored]
    # отчётность, из которой получаются те же коэффициенты покупателя, что в buyers.csv партии
    buyers = [{
        "inn": inn, "okved_section": row.okved_section, "region": row.region,
        "registration_date": f"{report_year[inn] - int(row.age_years)}-06-01",
        "financials": [{"year": int(report_year[inn]), "revenue": row.revenue_mln * 1e6, "short_term_liabilities": 1e6,
                        "cash": row.abs_liquidity * 1e6, "current_assets": row.current_liquidity * 1e6,
                        "cost_of_sales": 1e6, "payables": row.payables_days / 365 * 1e6}],
    } for inn, row in by.iterrows()]

    response = trained.post("/v1/delay/score", json={"calc_date": CALC_DATE, "delay_threshold_days": 14, "buyers": buyers,
                                                     "upds": request_upds, "history": history})

    assert response.status_code == 200
    assert response.json()["model_version"] == "delay-1.0"
    results = {item["upd_id"]: item for item in response.json()["results"]}
    for row in reference.itertuples():
        result = results[row.upd_id]
        if row.status == "no_history":
            assert result["status"] == "NO_PAYMENT_HISTORY"
            continue
        assert result["status"] == "SCORED"
        assert abs(result["p_delay"] - row.p_14) <= P_TOLERANCE + 1e-6
        assert abs(result["p_overdue_90"] - row.p_90) <= P_TOLERANCE + 1e-6
        assert abs(result["expected_delay_days"] - row.expected_days) <= DAYS_TOLERANCE + 1e-2


def test_delay_outside_model_domain_is_error(delay):
    """Модель определена для открытых УПД до срока оплаты: просроченный - ERROR, пачка не падает."""
    upd = {"number": "U-1", "client_inn": "C1", "client_industry": "TRADE", "buyer_inn": "B1", "buyer_pd": 0.01,
           "amount": 1e6, "outstanding": 1e6, "upd_date": "2026-08-01", "deferral_days": 30}
    request = main.DelayRequest(calc_date=date(2026, 10, 24), delay_threshold_days=14, upds=[
        {**upd, "upd_id": "1", "planned_due_date": "2026-09-01"},
        {**upd, "upd_id": "2", "planned_due_date": "2026-11-20"},
    ], history=[])

    first, second = delay.score(request)

    assert first.status == "ERROR" and "до срока оплаты" in first.error
    assert second.status == "NO_PAYMENT_HISTORY"


FINANCIALS = {"assets_total": 5.6e9, "current_assets": 3.3e9, "cash": 3.5e8, "equity": 2.35e9, "long_term_borrowings": 7.5e8,
              "short_term_liabilities": 2.1e9, "short_term_borrowings": 5.5e8, "receivables": 9e8, "payables": 8e8,
              "revenue": 7.9e9, "cost_of_sales": 6.7e9, "sales_profit": 7.2e8, "interest_payable": 8.5e7, "net_profit": 4.6e8}


def test_pd_model_and_base_pd(trained):
    strong = {"inn": "1", "okved_section": "F", "region": "Москва", "registration_date": "2008-04-10",
              "financials": [{**FINANCIALS, "year": 2025}, {**FINANCIALS, "year": 2024}, {**FINANCIALS, "year": 2023}]}
    weak = {**strong, "inn": "2", "financials": [{**FINANCIALS, "year": 2025, "equity": -1.2e8, "net_profit": -8e7, "cash": 3e6},
                                                 {**FINANCIALS, "year": 2024}]}
    small = {"inn": "3", "okved_section": "G", "financials": [{**FINANCIALS, "year": 2025, "revenue": 8e7}]}
    bank = {**strong, "inn": "4", "okved_section": "K"}

    response = trained.post("/v1/pd/score", json={"calc_date": "2026-10-24", "buyers": [strong, weak, small, bank]})

    assert response.status_code == 200
    assert response.json()["model_version"] == "pd-1.0"
    by_inn = {item["inn"]: item for item in response.json()["results"]}
    assert by_inn["1"]["pd_source"] == "MODEL" and by_inn["2"]["pd_source"] == "MODEL"
    assert by_inn["2"]["pd"] > by_inn["1"]["pd"]
    assert len(by_inn["2"]["reasons"]) == 3
    assert by_inn["2"]["reasons"][0]["direction"] == "INCREASES_RISK"
    assert by_inn["2"]["reasons"][0]["title"][0].isupper()
    assert by_inn["3"] == {**by_inn["3"], "pd_source": "BASE", "low_data": True, "low_data_reason": "LOW_REVENUE", "pd": 0.013}
    assert by_inn["4"]["low_data_reason"] == "EXCLUDED_INDUSTRY" and by_inn["4"]["pd_source"] == "BASE"


def test_models_endpoint_reports_trained_versions(trained):
    models = trained.get("/v1/models").json()

    assert models["pd"] == {**models["pd"], "version": "pd-1.0", "stub": False}
    assert models["delay"] == {**models["delay"], "version": "delay-1.0", "stub": False}


def test_rfsd_region():
    assert pd_trained.rfsd_region("Москва") == "moscow city"
    assert pd_trained.rfsd_region("Московская область") == "moscow reg."
    assert pd_trained.rfsd_region("Санкт-Петербург") == "sankt-petersburg"
    assert pd_trained.rfsd_region("bashkortostan") == "bashkortostan"
