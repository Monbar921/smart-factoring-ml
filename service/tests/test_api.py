"""Контракт сервиса моделей (contract/ml-service-openapi.yaml) и правила отсутствия данных п. 4.5."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

FINANCIALS_2025 = {
    "year": 2025, "assets_total": 5.6e9, "current_assets": 3.3e9, "cash": 3.5e8, "equity": 2.35e9,
    "long_term_borrowings": 7.5e8, "short_term_liabilities": 2.1e9, "short_term_borrowings": 5.5e8,
    "revenue": 7.9e9, "cost_of_sales": 6.7e9, "sales_profit": 7.2e8, "interest_payable": 8.5e7, "net_profit": 4.6e8,
}
WEAK_2025 = {**FINANCIALS_2025, "equity": -1.2e8, "net_profit": -8e7, "cash": 3e6}


def test_health_and_versions():
    assert client.get("/v1/health").json() == {"status": "UP"}
    models = client.get("/v1/models").json()
    assert models["pd"]["version"] == "pd-heuristic-1"
    assert models["supported_delay_thresholds"] == [7, 14, 30]


def test_pd_model_and_base_pd():
    response = client.post("/v1/pd/score", json={
        "calc_date": "2026-10-24",
        "buyers": [
            {"inn": "1", "okved_section": "F", "registration_date": "2008-04-10", "financials": [FINANCIALS_2025]},
            {"inn": "2", "okved_section": "F", "registration_date": "2008-04-10", "financials": [WEAK_2025]},
            {"inn": "3", "okved_section": "G", "financials": [{**FINANCIALS_2025, "revenue": 8e7}]},
            {"inn": "4", "okved_section": "C", "financials": [{**FINANCIALS_2025, "year": 2023}]},
        ],
    }).json()

    strong, weak, small, old = response["results"]
    assert strong["pd_source"] == "MODEL" and not strong["low_data"]
    assert weak["pd"] > strong["pd"]
    assert len(weak["reasons"]) == 3
    assert weak["reasons"][0]["direction"] == "INCREASES_RISK"
    assert small == {**small, "pd_source": "BASE", "low_data": True, "low_data_reason": "LOW_REVENUE", "pd": 0.03}
    assert old["low_data_reason"] == "NO_RECENT_REPORT"


def history(client_inn, buyer_inn, planned, paid=None, status="PAID"):
    return {"client_inn": client_inn, "buyer_inn": buyer_inn, "amount": 1e6, "upd_date": "2026-06-01",
            "planned_due_date": planned, "status": status, "paid_date": paid, "outstanding": 0 if paid else 1e6}


def upd(upd_id, client_inn, buyer_inn):
    return {"upd_id": upd_id, "number": f"U-{upd_id}", "client_inn": client_inn, "buyer_inn": buyer_inn, "buyer_pd": 0.02,
            "amount": 3e6, "outstanding": 3e6, "upd_date": "2026-10-01", "planned_due_date": "2026-11-20", "deferral_days": 50}


def test_delay_situations_of_missing_history():
    response = client.post("/v1/delay/score", json={
        "calc_date": "2026-10-24",
        "delay_threshold_days": 14,
        "upds": [upd("1", "C1", "B1"), upd("2", "C2", "B1"), upd("3", "C1", "B2"), upd("4", "C1", "B3")],
        "history": [
            history("C1", "B1", "2026-07-01", "2026-07-05"),
            history("C1", "B3", "2026-07-01", "2026-09-30"),
            history("C1", "B3", "2026-08-01", None, "OPEN"),
            history("C1", "B2", "2026-09-01", None, "OPEN"),
        ],
    }).json()

    on_time, other_pair, no_history, late = response["results"]
    assert on_time["status"] == "SCORED" and not on_time["no_pair_history"]
    assert other_pair["status"] == "SCORED" and other_pair["no_pair_history"]
    assert no_history["status"] == "NO_PAYMENT_HISTORY"
    assert late["p_delay"] > on_time["p_delay"]
    assert late["p_overdue_90"] > on_time["p_overdue_90"]
    assert "p_overdue_90" in on_time


def test_rejects_unsupported_threshold():
    response = client.post("/v1/delay/score", json={"calc_date": "2026-10-24", "delay_threshold_days": 10, "upds": []})
    assert response.status_code == 400
