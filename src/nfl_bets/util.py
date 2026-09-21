from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_utc(value: datetime | None = None) -> str:
    instant = value or utc_now()
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    return instant.astimezone(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, datetime):
        return iso_utc(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return int(numeric) if numeric.is_integer() else round(numeric, 12)
    if isinstance(value, dict):
        return {str(key): _hash_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hash_value(item) for item in value]
    return str(value)


def canonical_hash(record: dict[str, Any], excluded: set[str] | None = None) -> str:
    excluded = excluded or {"content_hash", "retrieved_at_utc"}
    normalized = {
        key: _hash_value(value) for key, value in record.items() if key not in excluded
    }
    payload = json.dumps(normalized, sort_keys=True, default=str, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def atomic_write_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def atomic_write_text(target: Path, text: str) -> None:
    atomic_write_bytes(target, text.encode("utf-8"))
