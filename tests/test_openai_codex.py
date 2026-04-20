"""Tests for OpenAI Codex OAuth-backed routing."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from nadirclaw.server import (
    ChatCompletionRequest,
    _build_openai_codex_tools,
    _call_litellm,
    _use_public_openai_responses_api,
    app,
)


class _MockResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    async def aiter_lines(self):
        yield "data: " + (
            '{"type":"response.completed","response":'
            + __import__("json").dumps(self._payload)
            + "}"
        )
        yield "data: [DONE]"


class _MockStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _MockAsyncClient:
    def __init__(self, response):
        self.response = response
        self.post_calls = []
        self.stream_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.response

    def stream(self, method, url, **kwargs):
        self.stream_calls.append((method, url, kwargs))
        return _MockStreamContext(self.response)


def test_openai_codex_tools_are_flattened():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    converted = _build_openai_codex_tools(tools)
    assert converted == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]


def test_openai_codex_endpoint_selection_uses_credential_source():
    assert _use_public_openai_responses_api("sk-test-key", "oauth") is False
    assert _use_public_openai_responses_api("not-an-sk-token", "openclaw") is False
    assert _use_public_openai_responses_api("not-an-sk-token", "env") is True
    assert _use_public_openai_responses_api("sk-test-key", "manual") is True


@pytest.mark.asyncio
async def test_openai_codex_uses_responses_api():
    request = ChatCompletionRequest(
        model="openai-codex/gpt-5.4",
        messages=[{"role": "user", "content": "Write a function"}],
        reasoning_effort="high",
    )
    response = _MockResponse(
        payload={
            "output_text": "def fn(): pass",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "def fn(): pass"}],
                }
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
                "output_tokens_details": {"reasoning_tokens": 3},
            },
        }
    )
    client = _MockAsyncClient(response)

    with patch("nadirclaw.credentials.get_credential", return_value="sk-test-key"), patch(
        "nadirclaw.credentials.get_credential_source",
        return_value="env",
    ), patch(
        "httpx.AsyncClient",
        return_value=client,
    ), patch("litellm.acompletion", new_callable=AsyncMock) as mock_comp:
        result = await _call_litellm("openai-codex/gpt-5.4", request, "openai-codex")

    assert mock_comp.await_count == 0
    assert result["content"] == "def fn(): pass"
    assert result["prompt_tokens"] == 12
    assert result["completion_tokens"] == 8
    assert result["reasoning_tokens"] == 3

    url, kwargs = client.post_calls[0]
    assert url == "https://api.openai.com/v1/responses"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-test-key"
    assert kwargs["json"]["model"] == "gpt-5.4"
    assert kwargs["json"]["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_openai_codex_oauth_uses_chatgpt_backend():
    request = ChatCompletionRequest(
        model="openai-codex/gpt-5.4",
        messages=[
            {"role": "system", "content": "Be terse and precise."},
            {"role": "user", "content": "Write a function"},
        ],
    )
    response = _MockResponse(
        payload={
            "output_text": "def fn(): pass",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "def fn(): pass"}],
                }
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8,
            },
        }
    )
    client = _MockAsyncClient(response)

    with patch("nadirclaw.credentials.get_credential", return_value="oauth-token"), patch(
        "nadirclaw.credentials.get_credential_source",
        return_value="oauth",
    ), patch(
        "httpx.AsyncClient",
        return_value=client,
    ):
        result = await _call_litellm("openai-codex/gpt-5.4", request, "openai-codex")

    assert result["content"] == "def fn(): pass"
    method, url, kwargs = client.stream_calls[0]
    assert method == "POST"
    assert url == "https://chatgpt.com/backend-api/codex/responses"
    assert kwargs["json"]["store"] is False
    assert kwargs["json"]["stream"] is True
    assert kwargs["json"]["instructions"] == "Be terse and precise."
    assert kwargs["json"]["input"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_openai_codex_oauth_instructions_default_when_missing():
    request = ChatCompletionRequest(
        model="openai-codex/gpt-5.4",
        messages=[{"role": "user", "content": "Write a function"}],
    )
    response = _MockResponse(
        payload={
            "output_text": "def fn(): pass",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "def fn(): pass"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    client = _MockAsyncClient(response)

    with patch("nadirclaw.credentials.get_credential", return_value="oauth-token"), patch(
        "nadirclaw.credentials.get_credential_source",
        return_value="oauth",
    ), patch(
        "httpx.AsyncClient",
        return_value=client,
    ):
        await _call_litellm("openai-codex/gpt-5.4", request, "openai-codex")

    _, _, kwargs = client.stream_calls[0]
    assert kwargs["json"]["instructions"]


def test_streaming_openai_codex_uses_batch_wrapper():
    client = TestClient(app)

    async def _fake_fallback(selected_model, request, provider, analysis_info):
        return (
            {
                "content": "batched codex response",
                "finish_reason": "stop",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            },
            selected_model,
            analysis_info,
        )

    with patch("nadirclaw.server._call_with_fallback", new=AsyncMock(side_effect=_fake_fallback)) as mock_fb, patch(
        "nadirclaw.server._stream_with_fallback",
        new=AsyncMock(),
    ) as mock_stream:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "openai-codex/gpt-5.4",
                "stream": True,
                "messages": [{"role": "user", "content": "Write code"}],
            },
        )

    assert resp.status_code == 200
    assert mock_fb.await_count == 1
    assert mock_stream.call_count == 0
    assert "text/event-stream" in resp.headers["content-type"]
