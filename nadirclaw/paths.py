"""Filesystem path helpers for configurable NadirClaw/OpenClaw homes."""

import os
from pathlib import Path


def _expand_path(raw: str) -> Path:
    return Path(os.path.expanduser(raw)).resolve()


def nadirclaw_home() -> Path:
    """Return the runtime home for NadirClaw state."""
    raw = os.getenv("NADIRCLAW_HOME") or os.getenv("NADIRCLAW_INSTALL_DIR") or "~/.nadirclaw"
    return _expand_path(raw)


def openclaw_home() -> Path:
    """Return the runtime home for OpenClaw state."""
    raw = os.getenv("OPENCLAW_HOME", "~/.openclaw")
    return _expand_path(raw)


def gemini_home() -> Path:
    """Return the runtime home for Gemini CLI state."""
    raw = os.getenv("GEMINI_HOME", "~/.gemini")
    return _expand_path(raw)


def codex_home() -> Path:
    """Return the runtime home for Codex state."""
    raw = os.getenv("CODEX_HOME", "~/.codex")
    return _expand_path(raw)


def continue_home() -> Path:
    """Return the runtime home for Continue state."""
    raw = os.getenv("CONTINUE_HOME", "~/.continue")
    return _expand_path(raw)


def cursor_home() -> Path:
    """Return the runtime home for Cursor state."""
    raw = os.getenv("CURSOR_HOME", "~/.cursor")
    return _expand_path(raw)


def nadirclaw_env_file() -> Path:
    return nadirclaw_home() / ".env"


def nadirclaw_credentials_path() -> Path:
    return nadirclaw_home() / "credentials.json"


def nadirclaw_log_dir() -> Path:
    raw = os.getenv("NADIRCLAW_LOG_DIR", str(nadirclaw_home() / "logs"))
    return _expand_path(raw)


def openclaw_auth_profiles_path() -> Path:
    return openclaw_home() / "agents" / "main" / "agent" / "auth-profiles.json"


def openclaw_legacy_config_path() -> Path:
    return openclaw_home() / "openclaw.json"
