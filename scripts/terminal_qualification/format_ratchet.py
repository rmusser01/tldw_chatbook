#!/usr/bin/env python3
"""Freeze and verify inherited Ruff formatter debt without editing source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Self, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from common import BoundedResult, QualificationError, run_bounded


SCHEMA_VERSION = 1
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@~^+\-]{0,254}\Z")


class RatchetError(RuntimeError):
    """Raised when the formatter ratchet cannot produce trustworthy evidence."""


class _FormatFactsEvidence(BaseModel):
    """Strict external representation of one file's formatter facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_sha256: StrictStr
    normalized_diff_sha256: StrictStr
    formatter_required: StrictBool
    debt_units: StrictInt = Field(ge=0)
    source_hunks: tuple[tuple[StrictInt, StrictInt], ...]

    @field_validator("source_sha256", "normalized_diff_sha256")
    @classmethod
    def _validate_digest(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("formatter digest must be lowercase SHA-256")
        return value

    @field_validator("source_hunks")
    @classmethod
    def _validate_hunks(
        cls, value: tuple[tuple[int, int], ...]
    ) -> tuple[tuple[int, int], ...]:
        if any(start < 0 or end < start for start, end in value):
            raise ValueError("formatter source hunk is invalid")
        return value


class _FormatBaselineEvidence(BaseModel):
    """Strict Pydantic boundary for the external formatter-baseline JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt
    generated_at_utc: StrictStr
    base_ref: StrictStr = Field(min_length=1, max_length=255)
    base_sha: StrictStr
    ruff_version: StrictStr = Field(min_length=1, max_length=128)
    paths: tuple[StrictStr, ...] = Field(min_length=1)
    baseline_red_paths: tuple[StrictStr, ...]
    files: dict[StrictStr, _FormatFactsEvidence]

    @field_validator("base_ref")
    @classmethod
    def _validate_base_ref(cls, value: str) -> str:
        if (
            GIT_REVISION_RE.fullmatch(value) is None
            or ".." in value
            or "@{" in value
            or "//" in value
        ):
            raise ValueError("formatter baseline base revision is invalid")
        return value

    @field_validator("base_sha")
    @classmethod
    def _validate_base_sha(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("formatter baseline base SHA is invalid")
        return value

    @field_validator("generated_at_utc")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("formatter baseline timestamp is invalid") from exc
        if timestamp.tzinfo is None:
            raise ValueError("formatter baseline timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def _validate_shape(self) -> Self:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported formatter baseline schema")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("formatter baseline paths must be unique")
        if set(self.files) != set(self.paths):
            raise ValueError("formatter baseline file set differs from paths")
        if not set(self.baseline_red_paths).issubset(self.paths):
            raise ValueError("formatter baseline red paths differ from paths")
        return self


class _RevisionInput(BaseModel):
    """Strict command-input boundary for one Git revision expression."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: StrictStr = Field(min_length=1, max_length=255)

    @field_validator("value")
    @classmethod
    def _validate_revision(cls, value: str) -> str:
        if (
            GIT_REVISION_RE.fullmatch(value) is None
            or ".." in value
            or "@{" in value
            or "//" in value
        ):
            raise ValueError("Git revision syntax is not allowed")
        return value


@dataclass(frozen=True)
class FormatFacts:
    """Content-free formatter facts for one source file."""

    source_sha256: str
    normalized_diff_sha256: str
    formatter_required: bool
    debt_units: int
    source_hunks: tuple[tuple[int, int], ...]

    def to_json(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_sha256,
            "normalized_diff_sha256": self.normalized_diff_sha256,
            "formatter_required": self.formatter_required,
            "debt_units": self.debt_units,
            "source_hunks": [list(item) for item in self.source_hunks],
        }


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    operation: str,
    timeout_seconds: float,
    input_bytes: bytes | None = None,
) -> BoundedResult:
    try:
        completed = run_bounded(
            argv,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            output_limit=16 * 1024 * 1024,
            operation=operation,
            input_bytes=input_bytes,
        )
    except QualificationError as exc:
        raise RatchetError(str(exc)) from exc
    if completed.timed_out:
        raise RatchetError(f"{operation} timed out")
    if completed.overflowed:
        raise RatchetError(f"{operation} exceeded output limit")
    return completed


def _git(repo: Path, *args: str) -> bytes:
    completed = _run(
        ("git", *args),
        cwd=repo,
        operation="formatter-git",
        timeout_seconds=30.0,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", "replace").strip()
        raise RatchetError(f"git {' '.join(args)} failed: {message}")
    return completed.stdout


def _repo_root() -> Path:
    completed = _run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=Path.cwd(),
        operation="formatter-repository-discovery",
        timeout_seconds=30.0,
    )
    if completed.returncode != 0:
        raise RatchetError("current directory is not inside a Git repository")
    return Path(completed.stdout.decode().strip()).resolve()


def _validated_revision(value: str, *, label: str) -> str:
    try:
        return _RevisionInput(value=value).value
    except ValidationError as exc:
        raise RatchetError(f"{label} revision is invalid") from exc


def _confined_repo_path(path: str | Path, repo: Path, *, label: str) -> Path:
    """Resolve one standalone-CLI path without importing the Chatbook app."""
    repo = repo.resolve()
    raw_path = Path(path)
    try:
        candidate = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (repo / raw_path).resolve()
        )
        candidate.relative_to(repo)
    except (OSError, ValueError) as exc:
        raise RatchetError(f"{label} path must stay inside the repository") from exc
    return candidate


def _validated_repo_path(path: Path, repo: Path, *, label: str) -> Path:
    validated = _confined_repo_path(path, repo, label=label)
    if validated.suffix.lower() != ".json":
        raise RatchetError(f"{label} path must name a JSON file")
    return validated


def _safe_paths(repo: Path, paths: Sequence[str]) -> tuple[str, ...]:
    if len(set(paths)) != len(paths):
        raise RatchetError("formatter paths must be unique")
    normalized: list[str] = []
    for raw in paths:
        path = PurePosixPath(raw)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise RatchetError(f"formatter path must be repository-relative: {raw}")
        relative = path.as_posix()
        validated = _confined_repo_path(
            relative,
            repo,
            label="formatter source",
        )
        validated_relative = validated.relative_to(repo).as_posix()
        if validated_relative != relative:
            raise RatchetError("formatter path must not traverse a symlink")
        normalized.append(relative)
    return tuple(normalized)


def _ruff_argv(repo: Path) -> tuple[str, ...]:
    argv = [sys.executable, "-m", "ruff"]
    config = repo / "pyproject.toml"
    if config.is_file():
        argv.extend(("--config", str(config)))
    return tuple(argv)


def _ruff_version(repo: Path) -> str:
    completed = _run(
        (*_ruff_argv(repo), "--version"),
        cwd=repo,
        operation="formatter-ruff-version",
        timeout_seconds=30.0,
    )
    if completed.returncode != 0:
        raise RatchetError("repository Ruff interpreter is unavailable")
    return completed.stdout.decode("utf-8", "replace").strip()


def _source_hunks(diff: str) -> tuple[tuple[int, int], ...]:
    hunks: list[tuple[int, int]] = []
    for line in diff.splitlines():
        match = HUNK_RE.match(line)
        if not match:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        hunks.append((start, start if count == 0 else start + count - 1))
    return tuple(hunks)


def _normalize_diff(diff: str, *, source_path: str, temp_root: Path) -> str:
    temp = temp_root.as_posix().rstrip("/")
    normalized_lines: list[str] = []
    for line in diff.replace("\\", "/").splitlines():
        line = line.replace(temp, "<TREE>")
        if line.startswith("--- "):
            line = f"--- source/{source_path}"
        elif line.startswith("+++ "):
            line = f"+++ formatted/{source_path}"
        elif line.startswith("@@ "):
            line = HUNK_RE.sub("@@ @@", line)
        normalized_lines.append(line.rstrip())
    return "\n".join(normalized_lines).strip() + ("\n" if normalized_lines else "")


def _format_facts(repo: Path, path: str, source: bytes, temp_root: Path) -> FormatFacts:
    materialized = temp_root / path
    materialized.parent.mkdir(parents=True, exist_ok=True)
    materialized.write_bytes(source)
    completed = _run(
        (
            *_ruff_argv(repo),
            "format",
            "--check",
            "--diff",
            "--no-cache",
            "--color",
            "never",
            str(materialized),
        ),
        cwd=repo,
        operation=f"formatter-check-{path}",
        timeout_seconds=60.0,
    )
    if completed.returncode not in (0, 1):
        message = completed.stderr.decode("utf-8", "replace").strip()
        raise RatchetError(f"Ruff failed for {path}: {message}")
    diff = completed.stdout.decode("utf-8", "replace")
    normalized = _normalize_diff(diff, source_path=path, temp_root=temp_root)
    debt_units = sum(
        1
        for line in normalized.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    )
    return FormatFacts(
        source_sha256=hashlib.sha256(source).hexdigest(),
        normalized_diff_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        formatter_required=bool(diff.strip()),
        debt_units=debt_units,
        source_hunks=_source_hunks(diff),
    )


def _base_sources(repo: Path, base_sha: str, paths: Sequence[str]) -> dict[str, bytes]:
    return {path: _git(repo, "show", f"{base_sha}:{path}") for path in paths}


def _worktree_sources(repo: Path, paths: Sequence[str]) -> dict[str, bytes]:
    sources: dict[str, bytes] = {}
    for path in paths:
        source_path = repo / path
        if not source_path.is_file():
            raise RatchetError(
                f"tracked formatter path is missing from the worktree: {path}"
            )
        sources[path] = source_path.read_bytes()
    return sources


def _measure(
    repo: Path,
    paths: Sequence[str],
    sources: dict[str, bytes],
) -> dict[str, FormatFacts]:
    with tempfile.TemporaryDirectory(prefix="tldw-format-ratchet-") as raw_temp:
        temp_root = Path(raw_temp)
        return {
            path: _format_facts(repo, path, sources[path], temp_root) for path in paths
        }


def _atomic_json(path: Path, payload: dict[str, object], *, replace: bool) -> None:
    if path.exists() and not replace:
        raise RatchetError(f"refusing to replace existing baseline: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def snapshot(*, base: str, output: Path, paths: Sequence[str], replace: bool) -> None:
    """Record immutable formatter facts for selected files at one Git base.

    Args:
        base: Validated Git revision to resolve to the immutable base commit.
        output: Repository-confined JSON destination for the baseline.
        paths: Unique repository-relative source paths to measure.
        replace: Whether an existing baseline may be atomically replaced.

    Returns:
        None.

    Raises:
        RatchetError: If input validation, Git resolution, formatting, or the
            atomic baseline write fails.
    """
    repo = _repo_root()
    output = _validated_repo_path(output, repo, label="formatter baseline output")
    safe_paths = _safe_paths(repo, paths)
    base = _validated_revision(base, label="base")
    base_sha = (
        _git(repo, "rev-parse", "--verify", f"{base}^{{commit}}").decode().strip()
    )
    facts = _measure(repo, safe_paths, _base_sources(repo, base_sha, safe_paths))
    payload = _FormatBaselineEvidence.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "base_ref": base,
            "base_sha": base_sha,
            "ruff_version": _ruff_version(repo),
            "paths": safe_paths,
            "baseline_red_paths": tuple(
                path for path in safe_paths if facts[path].formatter_required
            ),
            "files": {path: facts[path].to_json() for path in safe_paths},
        }
    )
    _atomic_json(output, payload.model_dump(mode="json"), replace=replace)


def _changed_ranges(
    repo: Path,
    base_sha: str,
    paths: Sequence[str],
    head_sha: str | None = None,
) -> dict[str, tuple[tuple[int, int], ...]]:
    revisions = (base_sha, head_sha) if head_sha is not None else (base_sha,)
    diff = _git(
        repo,
        "diff",
        "--unified=0",
        "--no-color",
        "--no-ext-diff",
        *revisions,
        "--",
        *paths,
    ).decode("utf-8", "replace")
    result: dict[str, list[tuple[int, int]]] = {path: [] for path in paths}
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            candidate = line[6:]
            current = candidate if candidate in result else None
            continue
        match = HUNK_RE.match(line)
        if current is None or not match:
            continue
        start = int(match.group(3))
        count = int(match.group(4) or "1")
        if count:
            result[current].append((start, start + count - 1))
    return {path: tuple(ranges) for path, ranges in result.items()}


def _overlaps(
    left: Sequence[tuple[int, int]],
    right: Sequence[tuple[int, int]],
) -> bool:
    return any(
        max(a_start, b_start) <= min(a_end, b_end)
        for a_start, a_end in left
        for b_start, b_end in right
    )


def verify(*, baseline: Path, head: str | None = None) -> None:
    """Verify immutable base facts and reject newly introduced formatter debt.

    Args:
        baseline: Repository-confined baseline JSON parsed by the strict model.
        head: Optional validated Git revision; when omitted, inspect the
            current worktree.

    Returns:
        None.

    Raises:
        RatchetError: If the baseline, revision, immutable facts, Ruff version,
            changed-line overlap, or normalized debt fails validation.
    """
    repo = _repo_root()
    baseline = _validated_repo_path(baseline, repo, label="formatter baseline")
    try:
        payload = _FormatBaselineEvidence.model_validate_json(
            baseline.read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise RatchetError(f"cannot read formatter baseline: {exc}") from exc
    except ValidationError as exc:
        raise RatchetError("formatter baseline validation failed") from exc
    base_sha = payload.base_sha
    baseline_files = payload.files
    safe_paths = _safe_paths(repo, payload.paths)
    recorded_ruff_version = payload.ruff_version
    current_ruff_version = _ruff_version(repo)
    if recorded_ruff_version != current_ruff_version:
        raise RatchetError(
            f"Ruff version drift: recorded {recorded_ruff_version!r}, "
            f"current {current_ruff_version!r}"
        )

    immutable_failures: list[str] = []
    if set(baseline_files) != set(safe_paths):
        immutable_failures.append("immutable base file set differs")
    base_facts = _measure(
        repo,
        safe_paths,
        _base_sources(repo, base_sha, safe_paths),
    )
    for path in safe_paths:
        stored = baseline_files.get(path)
        if (
            stored is None
            or stored.model_dump(mode="json") != base_facts[path].to_json()
        ):
            immutable_failures.append(f"{path}: immutable base facts differ")
    expected_red_paths = tuple(
        path for path in safe_paths if base_facts[path].formatter_required
    )
    if payload.baseline_red_paths != expected_red_paths:
        immutable_failures.append("immutable base red paths differ")
    if immutable_failures:
        raise RatchetError("\n".join(immutable_failures))

    if head is None:
        current_sources = _worktree_sources(repo, safe_paths)
        head_sha = None
    else:
        head = _validated_revision(head, label="head")
        head_sha = (
            _git(repo, "rev-parse", "--verify", f"{head}^{{commit}}").decode().strip()
        )
        current_sources = _base_sources(repo, head_sha, safe_paths)
    current = _measure(repo, safe_paths, current_sources)
    changed = _changed_ranges(repo, base_sha, safe_paths, head_sha)
    failures: list[str] = []
    for path in safe_paths:
        stored = baseline_files.get(path)
        if stored is None:
            failures.append(f"{path}: missing baseline facts")
            continue
        if _overlaps(current[path].source_hunks, changed[path]):
            failures.append(f"{path}: formatter-required hunk overlaps changed lines")
        baseline_units = stored.debt_units
        if current[path].debt_units > baseline_units:
            failures.append(
                f"{path}: normalized formatter debt grew from {baseline_units} "
                f"to {current[path].debt_units}"
            )
    if failures:
        raise RatchetError("\n".join(failures))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--base", required=True)
    snapshot_parser.add_argument("--output", required=True, type=Path)
    snapshot_parser.add_argument("--path", action="append", required=True)
    snapshot_parser.add_argument("--replace", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--head")
    verify_parser.add_argument("--baseline", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the formatter-ratchet command-line interface.

    Args:
        argv: Optional argument vector; defaults to process arguments.

    Returns:
        Zero on success or two when ratchet validation fails.
    """
    args = _parser().parse_args(argv)
    try:
        if args.command == "snapshot":
            snapshot(
                base=args.base,
                output=args.output,
                paths=args.path,
                replace=args.replace,
            )
        else:
            verify(baseline=args.baseline, head=args.head)
    except RatchetError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
