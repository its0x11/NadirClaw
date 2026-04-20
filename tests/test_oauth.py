"""Tests for nadirclaw.oauth — PKCE helpers, token validation, config resolution."""

import base64
import hashlib

import pytest

from nadirclaw.oauth import (
    _generate_code_challenge,
    _generate_code_verifier,
    validate_anthropic_setup_token,
)


class TestPKCE:
    def test_verifier_length(self):
        verifier = _generate_code_verifier()
        assert 43 <= len(verifier) <= 128

    def test_verifier_is_url_safe(self):
        verifier = _generate_code_verifier()
        # Should only contain URL-safe base64 characters (no padding)
        assert "=" not in verifier
        assert "+" not in verifier
        assert "/" not in verifier

    def test_challenge_matches_verifier(self):
        verifier = _generate_code_verifier()
        challenge = _generate_code_challenge(verifier)

        # Manually compute expected challenge
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        expected = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
        assert challenge == expected

    def test_different_verifiers_produce_different_challenges(self):
        v1 = _generate_code_verifier()
        v2 = _generate_code_verifier()
        assert v1 != v2
        assert _generate_code_challenge(v1) != _generate_code_challenge(v2)


class TestAnthropicSetupToken:
    def test_valid_token(self):
        token = "sk-ant-oat01-" + "x" * 80
        assert validate_anthropic_setup_token(token) is None

    def test_empty_token(self):
        error = validate_anthropic_setup_token("")
        assert error is not None
        assert "empty" in error.lower()

    def test_wrong_prefix(self):
        error = validate_anthropic_setup_token("sk-ant-wrong-" + "x" * 80)
        assert error is not None
        assert "sk-ant-oat01-" in error

    def test_too_short(self):
        error = validate_anthropic_setup_token("sk-ant-oat01-short")
        assert error is not None
        assert "short" in error.lower()

    def test_whitespace_trimmed(self):
        token = "  sk-ant-oat01-" + "x" * 80 + "  "
        assert validate_anthropic_setup_token(token) is None


class TestOpenAIOAuth:
    def test_openai_authorize_url_matches_codex_flow(self, monkeypatch):
        from nadirclaw.oauth import login_openai

        captured = {}

        class _Server:
            def shutdown(self):
                return None

        class _Queue:
            def get(self, timeout=None):
                return {"code": "auth-code", "state": captured["state"]}

        class _TokenResp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"access_token":"tok","refresh_token":"ref","expires_in":3600}'

        def _capture_open(url):
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            captured["url"] = url
            captured["state"] = params["state"][0]
            return True

        monkeypatch.setattr(
            "nadirclaw.oauth._start_callback_server",
            lambda timeout=300, port=1455, callback_path="/auth/callback": (_Server(), _Queue()),
        )
        monkeypatch.setattr("nadirclaw.oauth.webbrowser.open", _capture_open)
        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=30: _TokenResp())

        token_data = login_openai(timeout=1)

        assert token_data["access_token"] == "tok"
        assert "redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback" in captured["url"]
        assert "id_token_add_organizations=true" in captured["url"]
        assert "codex_cli_simplified_flow=true" in captured["url"]


class TestGeminiClientConfig:
    def test_env_var_override(self, monkeypatch):
        from nadirclaw.oauth import _resolve_gemini_client_config

        monkeypatch.setenv("NADIRCLAW_GEMINI_OAUTH_CLIENT_ID", "test-client-id")
        monkeypatch.setenv("NADIRCLAW_GEMINI_OAUTH_CLIENT_SECRET", "test-secret")

        config = _resolve_gemini_client_config()
        assert config["client_id"] == "test-client-id"
        assert config["client_secret"] == "test-secret"

    def test_no_gemini_cli_returns_empty(self, monkeypatch):
        from nadirclaw.oauth import _resolve_gemini_client_config

        # Clear all env vars
        for key in (
            "NADIRCLAW_GEMINI_OAUTH_CLIENT_ID",
            "OPENCLAW_GEMINI_OAUTH_CLIENT_ID",
            "GEMINI_CLI_OAUTH_CLIENT_ID",
        ):
            monkeypatch.delenv(key, raising=False)
        # Mock shutil.which to return None (no gemini CLI)
        monkeypatch.setattr("nadirclaw.oauth.shutil.which", lambda _: None)

        config = _resolve_gemini_client_config()
        assert config == {}
