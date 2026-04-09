"""Tests for configurable runtime path resolution."""

from pathlib import Path


def test_nadirclaw_home_uses_env(monkeypatch, tmp_path):
    from nadirclaw.paths import nadirclaw_credentials_path, nadirclaw_env_file, nadirclaw_home

    custom_home = tmp_path / "nadirclaw-test"
    monkeypatch.setenv("NADIRCLAW_HOME", str(custom_home))

    assert nadirclaw_home() == custom_home.resolve()
    assert nadirclaw_env_file() == custom_home.resolve() / ".env"
    assert nadirclaw_credentials_path() == custom_home.resolve() / "credentials.json"


def test_openclaw_home_uses_env(monkeypatch, tmp_path):
    from nadirclaw.paths import openclaw_auth_profiles_path, openclaw_home, openclaw_legacy_config_path

    custom_home = tmp_path / "openclaw-custom"
    monkeypatch.setenv("OPENCLAW_HOME", str(custom_home))

    assert openclaw_home() == custom_home.resolve()
    assert openclaw_auth_profiles_path() == (
        custom_home.resolve() / "agents" / "main" / "agent" / "auth-profiles.json"
    )
    assert openclaw_legacy_config_path() == custom_home.resolve() / "openclaw.json"


def test_other_tool_homes_use_env(monkeypatch, tmp_path):
    from nadirclaw.paths import codex_home, continue_home, cursor_home, gemini_home

    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("CONTINUE_HOME", str(tmp_path / "continue-home"))
    monkeypatch.setenv("CURSOR_HOME", str(tmp_path / "cursor-home"))
    monkeypatch.setenv("GEMINI_HOME", str(tmp_path / "gemini-home"))

    assert codex_home() == (tmp_path / "codex-home").resolve()
    assert continue_home() == (tmp_path / "continue-home").resolve()
    assert cursor_home() == (tmp_path / "cursor-home").resolve()
    assert gemini_home() == (tmp_path / "gemini-home").resolve()
