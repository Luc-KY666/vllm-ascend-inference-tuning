"""Stateless helpers for deployment report checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TextIO


def atomic_write_report(path: str | os.PathLike[str], report: Any) -> None:
    """Write a checkpoint using a same-directory temporary file and os.replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def tail_text(source: str | os.PathLike[str] | TextIO, limit: int) -> str:
    """Return at most ``limit`` trailing characters from a path or text stream."""

    if limit <= 0:
        return ""
    if hasattr(source, "read"):
        try:
            source.seek(0, os.SEEK_END)
            end = source.tell()
            source.seek(max(0, end - limit))
            value = source.read(limit)
        except (AttributeError, OSError, ValueError):
            value = source.read()[-limit:]
    else:
        candidate = Path(source)
        if candidate.is_file():
            with candidate.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - limit))
                value = stream.read(limit)
        else:
            value = str(source)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
