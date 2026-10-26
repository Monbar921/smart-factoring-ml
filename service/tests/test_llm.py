import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.llm import BackendLlm, LlmError, LlmUnavailable


class FakeBackend(BaseHTTPRequestHandler):
    """POST /internal/llm/v1/chat/completions как в backend: ответ задаётся в server.reply = (status, body)."""

    def do_POST(self):
        self.server.requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
        status, body = self.server.reply
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def backend():
    server = HTTPServer(("127.0.0.1", 0), FakeBackend)
    server.requests = []
    server.reply = (200, {})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def client(server) -> BackendLlm:
    return BackendLlm(f"http://127.0.0.1:{server.server_port}/internal/llm/v1", timeout_s=5)


def test_chat_goes_to_backend(backend):
    backend.reply = (200, {"content": "Риск низкий", "model": "GigaChat", "finishReason": "stop",
                           "promptTokens": 12, "completionTokens": 3})

    answer = client(backend).chat([{"role": "user", "content": "Оцени"}], temperature=0.2, max_tokens=100)

    assert answer.content == "Риск низкий"
    assert answer.model == "GigaChat"
    assert answer.finish_reason == "stop"
    assert (answer.prompt_tokens, answer.completion_tokens) == (12, 3)
    assert backend.requests == [("/internal/llm/v1/chat/completions",
                                 {"messages": [{"role": "user", "content": "Оцени"}], "temperature": 0.2, "maxTokens": 100})]


def test_disabled_llm_is_unavailable(backend):
    backend.reply = (503, {"code": "LLM_DISABLED", "message": "LLM выключена: LLM_MODE=stub", "details": {}})

    with pytest.raises(LlmUnavailable) as error:
        client(backend).chat([{"role": "user", "content": "Оцени"}])

    assert error.value.code == "LLM_DISABLED"


def test_invalid_request_is_error(backend):
    backend.reply = (400, {"code": "VALIDATION_ERROR", "message": "role", "details": {}})

    with pytest.raises(LlmError) as error:
        client(backend).chat([{"role": "tool", "content": "x"}])

    assert error.value.code == "VALIDATION_ERROR"
    assert not isinstance(error.value, LlmUnavailable)


def test_unreachable_backend_is_unavailable():
    with pytest.raises(LlmUnavailable) as error:
        BackendLlm("http://127.0.0.1:9/internal/llm/v1", timeout_s=2).chat([{"role": "user", "content": "x"}])

    assert error.value.code == "BACKEND_UNREACHABLE"
