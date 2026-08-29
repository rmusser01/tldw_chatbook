#!/usr/bin/env python3
"""Fail-closed redaction scan for the TASK-22868 evidence bundle."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys


_SENTINEL_ENV = "TASK22868_PRIVATE_SENTINEL"
_BUNDLE = Path(__file__).resolve().parent
_DEFAULT_TARGETS = (
    _BUNDLE,
    _BUNDLE.parents[2]
    / "UAT"
    / "2026-08-27-console-watchlists-workflow-round-trip.md",
)


def _patterns(sentinel: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
    slash = "/"
    fixture_body = r"\s+".join(("private", "article", "body"))
    return (
        ("private_sentinel", re.compile(re.escape(sentinel))),
        ("private_fixture_body", re.compile(fixture_body, re.IGNORECASE)),
        (
            "credential_assignment",
            re.compile(
                r"\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)"
                r"\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{8,}",
                re.IGNORECASE,
            ),
        ),
        (
            "authorization_or_cookie_header",
            re.compile(
                r"\b(?:authorization|proxy-authorization|cookie|set-cookie)"
                r"\b\s*[:=]\s*\S+",
                re.IGNORECASE,
            ),
        ),
        (
            "bearer_or_basic_payload",
            re.compile(r"\b(?:bearer|basic)\s+[A-Za-z0-9_./+=-]{8,}", re.IGNORECASE),
        ),
        (
            "real_home_path",
            re.compile(
                rf"{slash}(?:Users|home){slash}[A-Za-z0-9._-]+{slash}"
            ),
        ),
        (
            "concrete_temporary_profile_path",
            re.compile(
                rf"{slash}(?:private{slash}var{slash}folders|tmp){slash}[^\s<>'\"]+"
            ),
        ),
    )


def _files(targets: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        if not target.exists():
            raise FileNotFoundError(target)
        candidates = target.rglob("*") if target.is_dir() else (target,)
        for candidate in candidates:
            if candidate.is_symlink():
                raise RuntimeError(f"refusing symlink in scan scope: {candidate}")
            if candidate.is_file() and "__pycache__" not in candidate.parts:
                files.append(candidate)
    return sorted(set(files))


def main() -> int:
    sentinel = os.environ.get(_SENTINEL_ENV, "")
    if len(sentinel) < 16 or sentinel.isspace():
        print(
            f"ERROR: {_SENTINEL_ENV} must be supplied out of band "
            "and contain at least 16 characters.",
            file=sys.stderr,
        )
        return 2
    targets = tuple(Path(value).resolve() for value in sys.argv[1:]) or _DEFAULT_TARGETS
    try:
        files = _files(targets)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not files:
        print("ERROR: redaction scan scope contained no files.", file=sys.stderr)
        return 2

    findings: list[tuple[Path, int, str]] = []
    patterns = _patterns(sentinel)
    for path in files:
        text = path.read_bytes().decode("utf-8", errors="surrogateescape")
        for line_number, line in enumerate(text.splitlines(), 1):
            for pattern_name, pattern in patterns:
                if pattern.search(line):
                    findings.append((path, line_number, pattern_name))

    if findings:
        print(f"FAIL: {len(findings)} redaction finding(s) in {len(files)} files.")
        for path, line_number, pattern_name in findings:
            print(f"{path}:{line_number}: {pattern_name}")
        return 1
    print(f"PASS: 0 redaction findings in {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
