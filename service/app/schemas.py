"""Схемы API сервиса моделей. Контракт - contract/ml-service-openapi.yaml (копия из smart-factoring-backend)."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ModelVersion(BaseModel):
    version: str
    trained_at: date | None = None
    stub: bool = False


class ModelsInfo(BaseModel):
    pd: ModelVersion
    delay: ModelVersion
    supported_delay_thresholds: list[int]


class Financials(BaseModel):
    """Отчётность ГИР БО за год: строки форм в рублях."""

    year: int
    simplified: bool | None = None
    assets_total: float | None = None  # 1600
    current_assets: float | None = None  # 1200
    receivables: float | None = None  # 1230
    cash: float | None = None  # 1250
    equity: float | None = None  # 1300
    long_term_borrowings: float | None = None  # 1410
    short_term_liabilities: float | None = None  # 1500
    short_term_borrowings: float | None = None  # 1510
    payables: float | None = None  # 1520
    revenue: float | None = None  # 2110
    cost_of_sales: float | None = None  # 2120
    sales_profit: float | None = None  # 2200
    interest_payable: float | None = None  # 2330
    net_profit: float | None = None  # 2400


class Buyer(BaseModel):
    inn: str
    okved: str | None = None
    okved_section: str | None = None
    region: str | None = None
    registration_date: date | None = None
    financials: list[Financials] = Field(default_factory=list)


class PdRequest(BaseModel):
    calc_date: date
    buyers: list[Buyer]


class Reason(BaseModel):
    feature: str
    title: str
    value: float | None = None
    direction: Literal["INCREASES_RISK", "DECREASES_RISK"]
    contribution: float


class PdResult(BaseModel):
    inn: str
    status: Literal["SCORED", "ERROR"]
    pd: float | None = None
    pd_source: Literal["MODEL", "BASE"] | None = None
    low_data: bool = False
    low_data_reason: Literal["NO_RECENT_REPORT", "LOW_REVENUE", "YOUNG_COMPANY"] | None = None
    reasons: list[Reason] = Field(default_factory=list)
    error: str | None = None


class PdResponse(BaseModel):
    model_version: str
    results: list[PdResult]


class DelayUpd(BaseModel):
    upd_id: str
    number: str | None = None
    client_inn: str
    client_industry: str | None = None
    buyer_inn: str
    buyer_pd: float
    amount: float
    outstanding: float
    upd_date: date
    planned_due_date: date
    deferral_days: int


class HistoryUpd(BaseModel):
    client_inn: str
    buyer_inn: str
    amount: float
    upd_date: date
    planned_due_date: date
    status: Literal["PAID", "OPEN"]
    paid_date: date | None = None
    outstanding: float | None = None


class DelayRequest(BaseModel):
    calc_date: date
    delay_threshold_days: int  # 7, 14 или 30; иначе 400 (контракт)
    buyers: list[Buyer] = Field(default_factory=list)
    upds: list[DelayUpd]
    history: list[HistoryUpd] = Field(default_factory=list)


class DelayResult(BaseModel):
    upd_id: str
    status: Literal["SCORED", "NO_PAYMENT_HISTORY", "ERROR"]
    p_delay: float | None = None
    p_overdue_90: float | None = None
    expected_delay_days: float | None = None
    no_pair_history: bool = False
    error: str | None = None


class DelayResponse(BaseModel):
    model_version: str
    results: list[DelayResult]
