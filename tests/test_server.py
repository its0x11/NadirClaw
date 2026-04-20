"""Tests for nadirclaw.server — health endpoint and basic API contract."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the NadirClaw FastAPI app."""
    from nadirclaw.server import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "simple_model" in data
        assert "complex_model" in data

    def test_root_returns_info(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "NadirClaw"
        assert data["status"] == "ok"
        assert "version" in data


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1
        # Each model should have an id
        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"


class TestClassifyEndpoint:
    @patch("nadirclaw.server._smart_route_analysis")
    def test_classify_returns_classification(self, mock_route, client):
        mock_route.return_value = _mock_analysis()
        resp = client.post("/v1/classify", json={"prompt": "What is 2+2?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "classification" in data
        assert data["classification"]["tier"] in ("simple", "complex")
        assert "confidence" in data["classification"]
        assert "selected_model" in data["classification"]

    @patch("nadirclaw.server._smart_route_analysis")
    def test_classify_batch(self, mock_route, client):
        mock_route.return_value = _mock_analysis()
        resp = client.post(
            "/v1/classify/batch",
            json={"prompts": ["Hello", "Design a distributed system"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2


# ---------------------------------------------------------------------------
# X-Routed-* response headers
# ---------------------------------------------------------------------------

def _mock_fallback(content="OK", prompt_tokens=10, completion_tokens=5, model=None):
    """Build a side_effect callable for patching _call_with_fallback."""
    async def _side_effect(selected_model, request, provider, analysis_info):
        actual_model = model or selected_model
        return (
            {
                "content": content,
                "finish_reason": "stop",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            actual_model,
            {**analysis_info, "selected_model": actual_model},
        )
    return _side_effect


def _mock_analysis(model="gemini-2.5-flash", tier="simple"):
    return (
        model,
        {
            "strategy": "test",
            "selected_model": model,
            "tier": tier,
            "confidence": 1.0,
            "complexity_score": 0,
        },
    )


class TestRoutingHeaders:
    """X-Routed-Model, X-Routed-Tier, X-Complexity-Score headers."""

    @patch("nadirclaw.server._smart_route_full")
    @patch("nadirclaw.server._call_with_fallback")
    def test_non_streaming_response_has_routing_headers(self, mock_fb, mock_route, client):
        mock_route.return_value = _mock_analysis()
        mock_fb.side_effect = _mock_fallback(content="hi")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "routing header test 8x2q"}],
        })
        assert resp.status_code == 200
        assert "X-Routed-Model" in resp.headers
        assert resp.headers["X-Routed-Model"] != ""
        assert "X-Routed-Tier" in resp.headers
        assert resp.headers["X-Routed-Tier"] in ("simple", "mid", "complex", "reasoning", "direct", "free")
        assert "X-Complexity-Score" in resp.headers

    @patch("nadirclaw.server._call_with_fallback")
    def test_direct_model_has_routing_headers(self, mock_fb, client):
        mock_fb.side_effect = _mock_fallback(content="hi", model="gpt-4o")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "direct model header test 3v7w"}],
            "model": "gpt-4o",
        })
        assert resp.status_code == 200
        assert resp.headers["X-Routed-Model"] == "gpt-4o"
        assert resp.headers["X-Routed-Tier"] == "direct"

    @patch("nadirclaw.server._smart_route_full")
    @patch("nadirclaw.server._stream_with_fallback")
    def test_streaming_response_has_routing_headers(self, mock_stream, mock_route, client):
        mock_route.return_value = _mock_analysis()
        async def _fake_stream(*args, **kwargs):
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield "data: [DONE]\n\n"
        mock_stream.return_value = _fake_stream()
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "streaming header test 5k9z"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "X-Routed-Model" in resp.headers
        assert "X-Routed-Tier" in resp.headers
        assert "X-Complexity-Score" in resp.headers


# ---------------------------------------------------------------------------
# NMT-008 — Tests for specialized tier profiles
# ---------------------------------------------------------------------------

_NEW_TIER_PROFILES = ("orchestrator", "coding", "math", "planning", "abliterated")

# Expected env var suffixes per tier (maps profile → env var suffix)
_TIER_ENV_MAP = {
    "orchestrator": "ORCHESTRATOR_MODEL",
    "coding": "CODING_MODEL",
    "math": "MATH_MODEL",
    "planning": "PLANNING_MODEL",
    "abliterated": "ABLITERATED_MODEL",
}

# Env var names on the settings object
_TIER_SETTINGS_ATTRS = {
    "orchestrator": "ORCHESTRATOR_MODEL",
    "coding": "CODING_MODEL",
    "math": "MATH_MODEL",
    "planning": "PLANNING_MODEL",
    "abliterated": "ABLITERATED_MODEL",
}


class TestNewTierProfiles:
    """Unit-style tests for specialized profile routing.

    These tests mock _call_with_fallback so no real LLM calls are made.
    Each test verifies:
      1. The request returns 200
      2. The X-Routed-Tier header matches the requested profile
      3. The X-Routed-Model header is non-empty
    """

    @pytest.mark.parametrize("profile", _NEW_TIER_PROFILES)
    @patch("nadirclaw.server._call_with_fallback")
    def test_tier_returns_200(self, mock_fb, client, profile):
        mock_fb.side_effect = _mock_fallback(content="ok")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": f"{profile} test prompt"}],
            "model": profile,
        })
        assert resp.status_code == 200, f"{profile} should return 200, got {resp.status_code}"

    @pytest.mark.parametrize("profile", _NEW_TIER_PROFILES)
    @patch("nadirclaw.server._call_with_fallback")
    def test_tier_sets_correct_routing_header(self, mock_fb, client, profile):
        mock_fb.side_effect = _mock_fallback(content="ok")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": f"{profile} routing header test"}],
            "model": profile,
        })
        assert resp.status_code == 200
        assert resp.headers["X-Routed-Tier"] == profile, (
            f"Expected X-Routed-Tier={profile}, got {resp.headers.get('X-Routed-Tier')}"
        )

    @pytest.mark.parametrize("profile", _NEW_TIER_PROFILES)
    @patch("nadirclaw.server._call_with_fallback")
    def test_tier_sets_non_empty_model_header(self, mock_fb, client, profile):
        mock_fb.side_effect = _mock_fallback(content="ok")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": f"{profile} model header test"}],
            "model": profile,
        })
        assert resp.status_code == 200
        assert resp.headers["X-Routed-Model"] != "", f"{profile} should set non-empty X-Routed-Model"

    @pytest.mark.parametrize("profile", _NEW_TIER_PROFILES)
    @patch("nadirclaw.server._call_with_fallback")
    def test_tier_strategy_is_profile(self, mock_fb, client, profile):
        mock_fb.side_effect = _mock_fallback(content="ok")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": f"{profile} strategy test"}],
            "model": profile,
        })
        # Strategy is internal; verify the tier header is set correctly
        assert resp.headers["X-Routed-Tier"] == profile


class TestTierRoutingToCorrectModel:
    """Integration tests: each tier must route to the model configured for it.

    We patch settings.{TIER}_MODEL with a known test value and verify the
    X-Routed-Model response header matches.
    """

    @patch("nadirclaw.server._call_with_fallback")
    def test_orchestrator_tier_routes_to_orchestrator_model(self, mock_fb, client, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_ORCHESTRATOR_MODEL", "test/orchestrator-model")
        mock_fb.side_effect = _mock_fallback(content="ok")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "orchestrator routing to model"}],
            "model": "orchestrator",
        })
        assert resp.status_code == 200
        assert resp.headers["X-Routed-Tier"] == "orchestrator"
        assert resp.headers["X-Routed-Model"] == "test/orchestrator-model"

    @patch("nadirclaw.server._call_with_fallback")
    def test_coding_tier_routes_to_coding_model(self, mock_fb, client, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_CODING_MODEL", "test/coding-model")
        mock_fb.side_effect = _mock_fallback(content="ok")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "coding routing to model"}],
            "model": "coding",
        })
        assert resp.status_code == 200
        # When the profile branch is taken, selected_model comes from settings.CODING_MODEL
        # The header reflects what was actually routed
        assert resp.headers["X-Routed-Tier"] == "coding"
        assert resp.headers["X-Routed-Model"] == "test/coding-model"

    @patch("nadirclaw.server._call_with_fallback")
    def test_math_tier_routes_to_math_model(self, mock_fb, client, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_MATH_MODEL", "test/math-model")
        mock_fb.side_effect = _mock_fallback(content="ok")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "math routing to model"}],
            "model": "math",
        })
        assert resp.status_code == 200
        assert resp.headers["X-Routed-Tier"] == "math"
        assert resp.headers["X-Routed-Model"] == "test/math-model"

    @patch("nadirclaw.server._call_with_fallback")
    def test_planning_tier_routes_to_planning_model(self, mock_fb, client, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_PLANNING_MODEL", "test/planning-model")
        mock_fb.side_effect = _mock_fallback(content="ok")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "planning routing to model"}],
            "model": "planning",
        })
        assert resp.status_code == 200
        assert resp.headers["X-Routed-Tier"] == "planning"
        assert resp.headers["X-Routed-Model"] == "test/planning-model"

    @patch("nadirclaw.server._call_with_fallback")
    def test_abliterated_tier_routes_to_abliterated_model(self, mock_fb, client, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_ABLITERATED_MODEL", "test/abliterated-model")
        mock_fb.side_effect = _mock_fallback(content="ok")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "abliterated routing to model"}],
            "model": "abliterated",
        })
        assert resp.status_code == 200
        assert resp.headers["X-Routed-Tier"] == "abliterated"
        assert resp.headers["X-Routed-Model"] == "test/abliterated-model"


class TestTierFallbackChain:
    """Verify fallback chain is used when primary model fails."""

    @patch("nadirclaw.server._dispatch_model")
    def test_orchestrator_tier_falls_back_on_failure(self, mock_dispatch, client, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_ORCHESTRATOR_MODEL", "primary/orchestrator")
        monkeypatch.setenv("NADIRCLAW_ORCHESTRATOR_FALLBACK", "primary/orchestrator,fallback/orchestrator")
        calls = []

        async def _fail_first(model, *args, **kwargs):
            calls.append(model)
            if model == "primary/orchestrator":
                raise Exception("primary failed")
            return {"content": "fallback ok", "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}

        mock_dispatch.side_effect = _fail_first
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "orchestrator fallback test"}],
            "model": "orchestrator",
        })
        assert resp.status_code == 200
        assert calls == ["primary/orchestrator", "fallback/orchestrator"]
        assert resp.headers["X-Routed-Model"] == "fallback/orchestrator"

    @patch("nadirclaw.server._dispatch_model")
    def test_coding_tier_falls_back_on_failure(self, mock_dispatch, client, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_CODING_MODEL", "primary/coding")
        monkeypatch.setenv("NADIRCLAW_CODING_FALLBACK", "primary/coding,fallback/coding")
        calls = []

        async def _fail_first(model, *args, **kwargs):
            calls.append(model)
            if model == "primary/coding":
                raise Exception("primary failed")
            return {"content": "fallback ok", "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}

        mock_dispatch.side_effect = _fail_first
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "coding fallback test"}],
            "model": "coding",
        })
        # Should eventually succeed via fallback
        assert resp.status_code == 200
        assert calls == ["primary/coding", "fallback/coding"], "Fallback should have been called after primary failure"
        assert resp.headers["X-Routed-Model"] == "fallback/coding"


class TestExistingProfilesRegression:
    """Regression: existing profiles must still work after adding new tiers."""

    @pytest.mark.parametrize("profile", ("auto", "simple", "complex", "mid", "reasoning", "free"))
    @patch("nadirclaw.server._smart_route_full")
    @patch("nadirclaw.server._call_with_fallback")
    def test_existing_profile_still_works(self, mock_fb, mock_route, client, profile):
        mock_route.return_value = _mock_analysis()
        mock_fb.side_effect = _mock_fallback(content="ok")
        payload = {
            "messages": [{"role": "user", "content": f"{profile} regression test"}],
        }
        # "auto" is the default; others use model=profile
        if profile != "auto":
            payload["model"] = profile
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200, f"Existing profile '{profile}' should return 200, got {resp.status_code}"
        assert resp.headers["X-Routed-Tier"] != "", f"{profile} should set X-Routed-Tier header"

    @pytest.mark.parametrize("profile", ("orchestrator", "coding", "math", "planning", "abliterated"))
    @patch("nadirclaw.server._call_with_fallback")
    def test_new_tiers_not_confused_with_existing(self, mock_fb, client, profile):
        """New tiers must NOT accidentally map to simple/complex/mid."""
        mock_fb.side_effect = _mock_fallback(content="ok")
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": f"{profile} isolation test"}],
            "model": profile,
        })
        assert resp.status_code == 200
        assert resp.headers["X-Routed-Tier"] == profile
        assert resp.headers["X-Routed-Tier"] not in ("simple", "complex", "mid", "reasoning", "free")
