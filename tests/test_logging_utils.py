"""Tests for log rotation helpers."""

import gzip
import sqlite3


def test_prune_log_dir_compresses_legacy_logs(tmp_path):
    from nadirclaw.logging_utils import prune_log_dir

    legacy = tmp_path / "nadirclaw.err.log"
    legacy.write_text("x" * 300)

    result = prune_log_dir(
        tmp_path,
        default_max_bytes=128,
        default_backup_count=2,
        compress_current=True,
    )

    assert result["compressed"]
    assert legacy.exists()
    assert legacy.stat().st_size == 0
    assert list(tmp_path.glob("nadirclaw.err.log.*.gz"))


def test_request_jsonl_logger_rotates_and_compresses(tmp_path):
    from nadirclaw.logging_utils import get_request_jsonl_logger

    log_path = tmp_path / "requests.jsonl"
    logger = get_request_jsonl_logger(path=log_path, max_bytes=120, backup_count=2)

    for i in range(8):
        logger.info('{"n": %d, "message": "%s"}', i, "x" * 40)

    rotated = sorted(tmp_path.glob("requests.jsonl.*.gz"))
    assert rotated, "expected at least one rotated compressed request log"

    with gzip.open(rotated[0], "rt", encoding="utf-8") as handle:
        rotated_content = handle.read()
    assert '"message"' in rotated_content
    assert log_path.exists()
    assert log_path.stat().st_size <= 120


def test_vacuum_sqlite_db_prunes_old_rows(tmp_path):
    from nadirclaw.logging_utils import vacuum_sqlite_db
    from datetime import datetime, timedelta, timezone

    db_path = tmp_path / "requests.db"
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE requests (timestamp TEXT)")
        old_ts = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        new_ts = datetime.now(timezone.utc).isoformat()
        cursor.executemany("INSERT INTO requests (timestamp) VALUES (?)", [(old_ts,), (new_ts,)])
        conn.commit()
    finally:
        conn.close()

    result = vacuum_sqlite_db(
        db_path,
        prune_before=datetime.now(timezone.utc) - timedelta(days=30),
    )

    assert result["vacuumed"] is True
    assert result["rows_deleted"] == 1
