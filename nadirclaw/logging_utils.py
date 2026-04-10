"""Logging helpers with rotation and compression for NadirClaw."""

import gzip
import logging
import logging.handlers
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


_request_logger_lock = Lock()
_request_jsonl_logger: Optional[logging.Logger] = None
_request_jsonl_path: Optional[Path] = None
_TIMESTAMP_SUFFIX_RE = re.compile(r"\.\d{8}-\d{6}$")
_NUMERIC_SUFFIX_RE = re.compile(r"\.\d+$")


def _gzip_namer(default_name: str) -> str:
    return default_name + ".gz"


def _gzip_rotator(source: str, dest: str) -> None:
    with open(source, "rb") as src, gzip.open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(source)


def build_rotating_file_handler(
    path: Path,
    *,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
    level: int = logging.DEBUG,
) -> logging.Handler:
    """Build a rotating file handler with gzip compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.namer = _gzip_namer
    handler.rotator = _gzip_rotator
    return handler


def configure_logging(
    *,
    log_dir: Path,
    level: int,
    max_bytes: int,
    backup_count: int,
    console_mode: Optional[str] = None,
) -> Path:
    """Configure root logging for the server process."""
    log_path = log_dir / "server.log"
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    handlers: list[logging.Handler] = []
    if should_log_to_console(console_mode):
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(level)
        console.setFormatter(formatter)
        handlers.append(console)
    file_handler = build_rotating_file_handler(
        log_path,
        max_bytes=max_bytes,
        backup_count=backup_count,
        formatter=formatter,
        level=level,
    )
    handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore"):
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = True

    return log_path


def get_request_jsonl_logger(*, path: Path, max_bytes: int, backup_count: int) -> logging.Logger:
    """Return a dedicated rotated logger for JSONL request logs."""
    global _request_jsonl_logger, _request_jsonl_path

    with _request_logger_lock:
        if _request_jsonl_logger is not None and _request_jsonl_path == path:
            return _request_jsonl_logger

        logger = logging.getLogger("nadirclaw.requests_jsonl")
        logger.handlers = []
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(
            build_rotating_file_handler(
                path,
                max_bytes=max_bytes,
                backup_count=backup_count,
                formatter=logging.Formatter("%(message)s"),
                level=logging.INFO,
            )
        )
        _request_jsonl_logger = logger
        _request_jsonl_path = path
        return logger


def should_log_to_console(mode: Optional[str] = None) -> bool:
    """Decide whether process logs should also go to stdout."""
    raw = (mode or os.getenv("NADIRCLAW_LOG_TO_CONSOLE", "auto")).strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return sys.stdout.isatty() or sys.stderr.isatty()


def _base_log_name(name: str) -> str:
    if name.endswith(".gz"):
        name = name[:-3]
    if _NUMERIC_SUFFIX_RE.search(name):
        name = _NUMERIC_SUFFIX_RE.sub("", name)
    if _TIMESTAMP_SUFFIX_RE.search(name):
        name = _TIMESTAMP_SUFFIX_RE.sub("", name)
    return name


def _rotate_plain_log_file(path: Path) -> Optional[Path]:
    """Compress the current plain-text log file and recreate an empty file."""
    if not path.exists() or path.stat().st_size == 0:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.name}.{timestamp}.gz")
    temp_path = path.with_name(f"{path.name}.{timestamp}.tmp")
    path.rename(temp_path)
    try:
        with open(temp_path, "rb") as src, gzip.open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    path.touch()
    return dest


def _enforce_backup_limit(log_dir: Path, base_name: str, backup_count: int) -> int:
    """Delete rotated backups beyond the configured retention count."""
    rotated = sorted(
        (
            p for p in log_dir.iterdir()
            if p.is_file() and p.name.endswith(".gz") and _base_log_name(p.name) == base_name
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for old_path in rotated[backup_count:]:
        old_path.unlink(missing_ok=True)
        removed += 1
    return removed


def prune_log_dir(
    log_dir: Path,
    *,
    default_max_bytes: int,
    default_backup_count: int,
    file_limits: Optional[Dict[str, Dict[str, int]]] = None,
    compress_current: bool = True,
) -> Dict[str, Any]:
    """Compress oversized plain logs and enforce retention for rotated backups."""
    log_dir.mkdir(parents=True, exist_ok=True)
    file_limits = file_limits or {}
    compressed: list[str] = []
    removed = 0

    for path in sorted(log_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix in (".db", ".json"):
            continue
        if path.name.endswith(".tmp"):
            continue

        base_name = _base_log_name(path.name)
        limits = file_limits.get(base_name, {})
        max_bytes = limits.get("max_bytes", default_max_bytes)
        backup_count = limits.get("backup_count", default_backup_count)

        if compress_current and not path.name.endswith(".gz") and path.stat().st_size > max_bytes:
            rotated = _rotate_plain_log_file(path)
            if rotated:
                compressed.append(rotated.name)

        removed += _enforce_backup_limit(log_dir, base_name, backup_count)

    return {
        "log_dir": str(log_dir),
        "compressed": compressed,
        "removed": removed,
    }


def vacuum_sqlite_db(db_path: Path, *, prune_before: Optional[datetime] = None) -> Dict[str, Any]:
    """Vacuum a SQLite request log DB, optionally pruning old rows first."""
    result = {
        "db_path": str(db_path),
        "rows_deleted": 0,
        "vacuumed": False,
    }
    if not db_path.exists():
        return result

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        if prune_before is not None:
            cursor.execute(
                "DELETE FROM requests WHERE timestamp < ?",
                (prune_before.astimezone(timezone.utc).isoformat(),),
            )
            result["rows_deleted"] = max(0, cursor.rowcount)
            conn.commit()
        cursor.execute("VACUUM")
        cursor.execute("ANALYZE")
        conn.commit()
        result["vacuumed"] = True
    finally:
        conn.close()
    return result
