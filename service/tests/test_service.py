"""Эксплуатация сервиса: загрузка моделей при старте, предел пачки, X-Request-ID."""

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.models import registry
from app.settings import Settings


@pytest.fixture
def fresh_models():
    registry.pd_model.cache_clear()
    registry.delay_model.cache_clear()
    yield
    registry.pd_model.cache_clear()
    registry.delay_model.cache_clear()


def test_models_are_loaded_on_startup(fresh_models):
    with TestClient(main.app) as client:
        assert registry.pd_model.cache_info().currsize == 1
        assert registry.delay_model.cache_info().currsize == 1
        assert client.get("/v1/health").json()["status"] == "UP"


def test_broken_model_module_fails_startup(fresh_models, monkeypatch):
    monkeypatch.setenv("PD_MODEL_MODULE", "app.models.no_such_model")

    with pytest.raises(ModuleNotFoundError):
        with TestClient(main.app):
            pass


def test_batch_limit(monkeypatch):
    monkeypatch.setattr(main, "settings", Settings(log_level="INFO", max_batch=2))
    client = TestClient(main.app)
    buyers = [{"inn": str(i)} for i in range(3)]

    response = client.post("/v1/pd/score", json={"calc_date": "2026-10-24", "buyers": buyers})

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "BATCH_TOO_LARGE"
    assert client.post("/v1/pd/score", json={"calc_date": "2026-10-24", "buyers": buyers[:2]}).status_code == 200


def test_request_id_is_echoed_or_generated():
    client = TestClient(main.app)

    assert client.get("/v1/models", headers={"X-Request-ID": "run-7"}).headers["X-Request-ID"] == "run-7"
    assert len(client.get("/v1/models").headers["X-Request-ID"]) == 16
