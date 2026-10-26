"""LLM для моделей - только через backend (Р-48).

Сервис моделей не ходит в GigaChat напрямую: ключ, токен, предел одновременных запросов и выбор модели - в backend
(POST /internal/rest/llm/chat/completions - backend напрямую по сети стенда, не через gateway).

    from app.llm import backend_llm, LlmUnavailable

    try:
        answer = backend_llm().chat([{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
                                    temperature=0.2, max_tokens=300)
        text = answer.content
    except LlmUnavailable:
        text = None  # LLM выключена (LLM_MODE в backend не gigachat) или недоступна - работаем без неё

Адрес - BACKEND_LLM_URL, по умолчанию http://backend:8080/internal/rest/llm.
"""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_URL = "http://backend:8080/internal/rest/llm"
# GigaChat с повторами в backend может отвечать десятки секунд
DEFAULT_TIMEOUT_S = 120


class LlmError(Exception):
    """Ошибка LLM: code из ErrorDto backend (VALIDATION_ERROR, LLM_ERROR...)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class LlmUnavailable(LlmError):
    """LLM выключена (LLM_DISABLED), недоступна (LLM_UNAVAILABLE) или нет связи с backend - стоит работать без LLM."""


@dataclass(frozen=True)
class LlmAnswer:
    content: str
    model: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int


class BackendLlm:
    def __init__(self, base_url: str = DEFAULT_URL, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def chat(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> LlmAnswer:
        """messages - [{"role": "system" | "user" | "assistant", "content": "..."}], по порядку диалога."""
        body = {"messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["maxTokens"] = max_tokens
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise _error(e.code, e.read()) from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise LlmUnavailable("BACKEND_UNREACHABLE", str(e)) from e
        return LlmAnswer(
            content=data["content"],
            model=data.get("model"),
            finish_reason=data.get("finishReason"),
            prompt_tokens=data.get("promptTokens", 0),
            completion_tokens=data.get("completionTokens", 0),
        )


def _error(status: int, body: bytes) -> LlmError:
    try:
        error = json.loads(body.decode("utf-8"))
        code, message = error.get("code", f"HTTP_{status}"), error.get("message", "")
    except ValueError:
        code, message = f"HTTP_{status}", body.decode("utf-8", "replace")[:300]
    if status == 503:
        return LlmUnavailable(code, message)
    return LlmError(code, message)


@lru_cache
def backend_llm() -> BackendLlm:
    return BackendLlm(os.getenv("BACKEND_LLM_URL", DEFAULT_URL))
