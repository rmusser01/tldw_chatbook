"""Cost bounds for Agent Change Review (TASK-1975).

Keeps the tracking substrate bounded, per the spec's cost posture:

* **Root budgets** — a root over ``max_files``/``max_total_bytes`` gets
  tracking DISABLED with honest copy, never a silent half-track.
* **Oversize excludes** — git cannot exclude by size, so files over
  ``max_file_bytes`` are appended to the shadow repo's ``info/exclude``
  dynamically at snapshot time and the review discloses the count.
* **Knobs** — read from the FLAT ``[change_review]`` config section (the
  dotted ``get_cli_setting`` form has dropped defaults before — task
  1754), each overridable via ``TLDW_CHANGE_REVIEW_<KEY>`` env vars.

The scan never follows symlinks (a link to a huge external tree must not
disqualify the root) and prunes the same directories the shadow repo's
forced excludes skip — those trees are never tracked, so they never count
against the budget either.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from tldw_chatbook.Workspaces.change_review_consent import (
    ChangeReviewCapability,
    ChangeReviewState,
)

DEFAULT_MAX_FILES = 20_000
DEFAULT_MAX_TOTAL_BYTES = 2 * 1024**3
DEFAULT_MAX_FILE_BYTES = 10 * 1024**2
DEFAULT_RETENTION_DAYS = 30
DEFAULT_MAX_SUB_ROOTS = 20


def _change_review_enabled_setting() -> object:
    """Read the global capability setting, preserving invalid values."""
    from tldw_chatbook.config import get_cli_setting

    return get_cli_setting("change_review", "enabled", True)


def _coerce_change_review_enabled(value: object) -> bool:
    """Strictly coerce one supported Boolean representation.

    Raises:
        ValueError: If ``value`` is not a supported Boolean representation.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError("unsupported Change Review capability value")


def read_change_review_capability() -> ChangeReviewCapability:
    """Read the global master capability without failing tracking open."""
    try:
        raw = os.environ.get("TLDW_CHANGE_REVIEW_ENABLED")
        if raw is None:
            raw = _change_review_enabled_setting()
        enabled = _coerce_change_review_enabled(raw)
    except Exception:  # noqa: BLE001 -- invalid config is unavailable
        logger.debug("change_review: global capability unavailable")
        return ChangeReviewCapability(ChangeReviewState.UNAVAILABLE)
    return ChangeReviewCapability(
        ChangeReviewState.ENABLED if enabled else ChangeReviewState.DISABLED
    )


def change_review_enabled_globally() -> bool:
    """Whether the global Change Review capability is explicitly available.

    Returns:
        True only for an enabled capability. Disabled and unreadable state
        both fail runtime tracking off.
    """
    return read_change_review_capability().state is ChangeReviewState.ENABLED

#: Directory names the scan prunes — mirrors the shadow repo's
#: ``FORCED_EXCLUDES`` (change_tracking.py): untracked trees must not count.
_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
    }
)


def _config_setting(key: str, default: int) -> int:
    """Read one knob from the flat ``[change_review]`` config section."""
    try:
        from tldw_chatbook.config import get_cli_setting

        return get_cli_setting("change_review", key, default)
    except Exception:  # noqa: BLE001 -- a broken config never breaks bounds
        return default


def change_review_setting(key: str, default: int) -> int:
    """Resolve a ``[change_review]`` knob: env var, then config, then default.

    Args:
        key: Flat-section key, e.g. ``"max_files"``.
        default: Value when neither env nor config provides one.

    Returns:
        The resolved integer; unparseable values fall back to ``default``.
    """
    env = os.environ.get(f"TLDW_CHANGE_REVIEW_{key.upper()}")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            logger.warning(
                "change_review: ignoring unparseable env override "
                f"TLDW_CHANGE_REVIEW_{key.upper()}={env!r}"
            )
    try:
        return int(_config_setting(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RootScan:
    """One root's budget scan result.

    Attributes:
        files: Files counted before finishing or aborting.
        total_bytes: Bytes counted before finishing or aborting.
        over_budget: True when either cap was crossed (counts are then
            partial — the scan aborts early rather than walking a tree it
            already knows is too big).
        oversized: Root-relative paths of files over ``max_file_bytes``,
            in walk order.
        nested_repos: Root-relative paths of NESTED git repos (a child
            directory carrying ``.git`` as dir or file — TASK-1976: git
            records these as gitlinks, so changes inside are invisible to
            snapshots; the hole must be disclosed). The root's own
            ``.git`` is not nested.
    """

    files: int
    total_bytes: int
    over_budget: bool
    oversized: tuple[str, ...] = ()
    nested_repos: tuple[str, ...] = ()


def scan_root(
    root: Path | str,
    *,
    max_files: int | None = None,
    max_total_bytes: int | None = None,
    max_file_bytes: int | None = None,
) -> RootScan:
    """Walk a root, counting files/bytes and collecting oversize paths.

    Args:
        root: The directory to scan.
        max_files: File-count budget; ``None`` reads the knob.
        max_total_bytes: Byte budget; ``None`` reads the knob.
        max_file_bytes: Per-file cap for the oversize list; ``None`` reads
            the knob.

    Returns:
        The scan result; unreadable entries are skipped (a permission
        error on one file must not disable tracking for the root).
    """
    if max_files is None:
        max_files = change_review_setting("max_files", DEFAULT_MAX_FILES)
    if max_total_bytes is None:
        max_total_bytes = change_review_setting(
            "max_total_bytes", DEFAULT_MAX_TOTAL_BYTES
        )
    if max_file_bytes is None:
        max_file_bytes = change_review_setting(
            "max_file_bytes", DEFAULT_MAX_FILE_BYTES
        )
    root = Path(root)
    files = 0
    total = 0
    oversized: list[str] = []
    nested: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Nested-repo detection (TASK-1976) BEFORE pruning: a child dir
        # carrying `.git` (dir or worktree FILE) is a disclosed hole. The
        # root's own `.git` is the root being a repo, not a nested one.
        if Path(dirpath) != root and (".git" in dirnames or ".git" in filenames):
            try:
                nested.append(Path(dirpath).relative_to(root).as_posix())
            except ValueError:  # pragma: no cover -- walk stays inside
                pass
            # Qodo #1254 finding 4: nothing under a nested repo is ever
            # trackable, so none of it may count against the root's budget
            # or pollute the oversize disclosure -- do not descend, do not
            # count its files.
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                if path.is_symlink():
                    continue
                size = path.stat().st_size
            except OSError:
                continue
            files += 1
            total += size
            if size > max_file_bytes:
                try:
                    oversized.append(path.relative_to(root).as_posix())
                except ValueError:  # pragma: no cover -- walk stays inside
                    continue
            if files > max_files or total > max_total_bytes:
                return RootScan(
                    files=files,
                    total_bytes=total,
                    over_budget=True,
                    oversized=tuple(oversized),
                    nested_repos=tuple(nested),
                )
    return RootScan(
        files=files,
        total_bytes=total,
        over_budget=False,
        oversized=tuple(oversized),
        nested_repos=tuple(nested),
    )
