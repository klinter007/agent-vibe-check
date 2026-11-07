"""Hermetic tests for the OpenAI-compatible provider against a tiny local fake server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_vibe_check.providers import ProviderError, build_provider, get_provider_info
from agent_vibe_check.providers.openai_compat import HTTPProviderError, OpenAICompatProvider
from agent_vibe_check.pipeline.llm_utils import _classify_error


class _Fake(BaseHTTPRequestHandler):
    calls: list[dict] = []
    reject_json_mode = False
    status = 200

    def log_message(self, *a):  # silence
        pass

    def _send(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"data": [{"id": "big-model"}, {"id": "text-embedding-3"}, {"id": "gpt-image-2"},
                                  {"id": "small-model"}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.calls.append({"payload": payload, "auth": self.headers.get("Authorization")})
        if _Fake.status != 200:
            return self._send(_Fake.status, {"error": "slow down"}, {"Retry-After": "7"})
        if _Fake.reject_json_mode and "response_format" in payload:
            return self._send(400, {"error": {"message": "response_format is not supported"}})
        self._send(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})


@pytest.fixture
def server():
    _Fake.calls, _Fake.reject_json_mode, _Fake.status = [], False, 200
    httpd = HTTPServer(("127.0.0.1", 0), _Fake)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def _provider(url, key="k", model="big-model"):
    return OpenAICompatProvider(get_provider_info("custom"), api_key=key, base_url=url, model=model)


def test_analyze_sends_system_json_mode_and_key(server):
    out = _provider(server).analyze("hi", system="RULES")
    assert out == '{"ok": true}'
    call = _Fake.calls[0]
    assert call["auth"] == "Bearer k"
    assert call["payload"]["model"] == "big-model"
    assert call["payload"]["response_format"] == {"type": "json_object"}
    assert "RULES" in call["payload"]["messages"][0]["content"]


def test_json_mode_fallback_when_server_rejects_it(server):
    _Fake.reject_json_mode = True
    p = _provider(server)
    assert p.analyze("hi") == '{"ok": true}'
    assert "response_format" not in _Fake.calls[-1]["payload"]


def test_list_models_filters_non_chat(server):
    assert _provider(server).list_models() == ["big-model", "small-model"]


def test_no_model_is_a_clear_error(server):
    with pytest.raises(ProviderError, match="No model chosen"):
        _provider(server, model=None).analyze("hi")


def test_rate_limit_is_classified_with_retry_after(server):
    _Fake.status = 429
    with pytest.raises(HTTPProviderError) as ei:
        _provider(server).analyze("hi")
    category, wait = _classify_error(ei.value)
    assert category == "rate_limit" and wait == 7.0


def test_local_provider_needs_no_key(monkeypatch, server):
    monkeypatch.setenv("OLLAMA_BASE_URL", server)
    p = build_provider("ollama")
    assert p.list_models() == ["big-model", "small-model"]


def test_unreachable_server_is_a_provider_error():
    with pytest.raises(ProviderError, match="cannot reach"):
        _provider("http://127.0.0.1:9/v1").list_models()
