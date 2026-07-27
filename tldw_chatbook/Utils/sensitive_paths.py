# tldw_chatbook/Utils/sensitive_paths.py
"""Paths refused by the agent-facing file tools, regardless of configured root.

Enforced directly inside ``Tools/file_operation_tools.py``'s ``ReadFileTool``,
``WriteFileTool``, ``ListDirectoryTool``, ``GlobFiles`` and ``GrepFiles`` --
each calls :func:`is_sensitive_path` (directly, or through that module's
``is_within`` helper) on every candidate path immediately after path
validation and before touching the filesystem. It is *not* wired into
``Utils/path_validation.validate_path``/``validate_path_multi`` themselves:
those helpers are the app's general-purpose validators, used by ~40
first-party call sites (config screens, DB path resolution, exports, ...) to
validate paths to this application's own config and database files -- which
are exactly the paths this module refuses. Baking the check in there would
block legitimate first-party access; it belongs at the agent-tool boundary
instead, shared by every file tool, so they cannot drift from each other.

Two distinct reasons a path lands here:

1. **Credentials.** ``read_file`` carries no elevated risk beyond ``reads``,
   so an unconfined read is a path from a private key into a persisted
   transcript that may be sent to any provider.
2. **This application's own gate state and data.** A tool able to rewrite
   ``mcp_permissions.json`` or ``config.toml`` can turn every ``ask`` into
   ``allow`` -- a one-step bypass of the permission system. A tool able to
   read or rewrite this app's own SQLite databases can exfiltrate or
   corrupt every conversation, note and credential-adjacent record they
   hold, bypassing the application layer entirely.

This is a guardrail, not a security boundary: it stops accidents and naive
injected payloads, not a determined ``python -c``. The sandbox/workspace-root
track is the real answer for shell execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from loguru import logger

#: Directory prefixes that are refused along with everything beneath them.
_SENSITIVE_DIRS = (
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    "~/.config/gcloud",
    "~/.docker",
    "~/.kube",
    "~/.local/share/keyrings",
)

#: Individual files that are refused.
_SENSITIVE_FILES = (
    "~/.config/tldw_cli/config.toml",
    "~/.config/tldw_cli/mcp_permissions.json",
)

#: Names of the ``config`` accessors for this app's own SQLite databases.
#: Called lazily (see ``_sensitive_db_paths``) rather than imported at module
#: scope: ``config`` also honors ``[database] *_db_path`` overrides and a
#: per-test ``HOME``, so the real path can only be known at call time -- and
#: importing a large, slow module at ``Utils`` import time is itself a cost
#: worth avoiding when most callers never need it.
_DB_PATH_ACCESSOR_NAMES = (
    "get_chachanotes_db_path",
    "get_prompts_db_path",
    "get_media_db_path",
    "get_library_collections_db_path",
    "get_library_ingest_jobs_db_path",
    "get_workspaces_db_path",
    "get_subscriptions_db_path",
    "get_notifications_db_path",
    "get_research_db_path",
    "get_writing_db_path",
    "get_scheduled_tasks_db_path",
)

#: Suffixes SQLite appends to a database's own filename for its sidecar
#: files: ``-wal``/``-shm`` under ``PRAGMA journal_mode=WAL`` (several of
#: this app's databases run in WAL mode) and ``-journal`` under the default
#: rollback-journal mode. Each sidecar holds the same class of recent data
#: as the database itself, so refusing only the ``.db`` path leaves them
#: readable the moment a sandbox/workspace root is widened to contain the
#: user data directory -- exactly the misconfiguration the DB denial exists
#: to guard against. Matching is exact-equality against a name built from
#: each enumerated DB's own filename (see ``_db_sidecar_paths``), never a
#: loose prefix: a file that merely *starts with* a DB's name (e.g.
#: ``chachanotes.db.backup-2026`` or ``chachanotes.db2``) is a different
#: file and is not matched by this.
_DB_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _resolved(path_str: str) -> Path | None:
    try:
        return Path(path_str).expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def _sensitive_db_paths() -> tuple[Path, ...]:
    """Resolve this app's own SQLite database paths, lazily.

    These databases live under ``config.get_user_data_dir()`` -- by default
    a sibling of ``~/.config/tldw_cli`` (e.g. ``~/.local/share/tldw_cli/...``),
    not beneath it, so the static ``_SENSITIVE_FILES``/``_SENSITIVE_DIRS``
    tuples above cannot express their location. Each path is resolved via
    the app's own accessor (which also honors ``[database]`` path
    overrides and the active user folder) rather than hardcoded, since
    neither the user folder nor an override is known statically.

    Returns:
        Resolved paths to every database whose accessor could be called.
        An accessor that raises is skipped rather than failing the whole
        check -- it is additional coverage, not the primary guarantee.
    """
    from .. import config as _config

    resolved: list[Path] = []
    for accessor_name in _DB_PATH_ACCESSOR_NAMES:
        accessor = getattr(_config, accessor_name, None)
        if accessor is None:
            continue
        try:
            resolved.append(accessor())
        except Exception as exc:  # noqa: BLE001 - defensive, additive coverage only
            logger.debug(
                f"sensitive_paths: could not resolve {accessor_name}: {exc}"
            )
    return tuple(resolved)


def _db_sidecar_paths(db_path: Path) -> tuple[Path, ...]:
    """Build the WAL/SHM/rollback-journal sidecar paths for one DB path.

    Args:
        db_path: A resolved path to one of this app's SQLite databases, as
            returned by ``_sensitive_db_paths()``.

    Returns:
        One path per entry in ``_DB_SIDECAR_SUFFIXES``, each formed by
        appending the suffix to ``db_path``'s own filename -- e.g.
        ``chachanotes.db`` -> ``chachanotes.db-wal``. Built from an explicit
        name construction, not a prefix, so callers must compare by exact
        equality: appending is not the same as matching anything that
        merely starts with the DB's name.
    """
    return tuple(db_path.with_name(db_path.name + suffix) for suffix in _DB_SIDECAR_SUFFIXES)


class SensitivePathContext(NamedTuple):
    """A snapshot of the resolved sensitive-path set, valid for one tool call.

    Building one of these costs the same 11 config-accessor resolutions
    ``is_sensitive_path`` would otherwise repeat on every invocation. A
    caller that tests many candidate paths within a single tool invocation
    (``GlobFiles``/``GrepFiles`` and ``ListDirectoryTool``'s recursive walk,
    all in ``Tools/file_operation_tools.py``) should build exactly ONE of
    these at the start of that invocation and pass it into every
    ``is_sensitive_path``/``is_within`` call it makes, rather than let each
    call re-resolve the set from scratch.

    Deliberately not cached at module or process scope -- see
    ``resolve_sensitive_context``.
    """

    files: tuple[Path, ...]
    dirs: tuple[Path, ...]
    db_paths: tuple[Path, ...]


def resolve_sensitive_context() -> SensitivePathContext:
    """Resolve the full sensitive-path set once, for reuse across many checks.

    Call this ONCE per tool invocation and thread the result through to
    every ``is_sensitive_path``/``is_within`` call that invocation makes.
    Do NOT cache the return value at module or process scope: the whole
    point of the per-call ``_sensitive_db_paths()`` resolution it wraps is
    to observe a config change (e.g. the test suite swapping
    ``TLDW_CONFIG_PATH`` between cases) on the very next call rather than
    serving a stale answer. A single invocation resolving this once is
    "per call"; a global cache would not be.

    Returns:
        A ``SensitivePathContext`` snapshotting the currently configured
        sensitive files, directories, and database paths (entries that
        failed to resolve are dropped).
    """
    return SensitivePathContext(
        files=tuple(
            p for p in (_resolved(entry) for entry in _SENSITIVE_FILES) if p is not None
        ),
        dirs=tuple(
            p for p in (_resolved(entry) for entry in _SENSITIVE_DIRS) if p is not None
        ),
        db_paths=tuple(
            p
            for p in (_resolved(str(raw)) for raw in _sensitive_db_paths())
            if p is not None
        ),
    )


def is_sensitive_path(
    candidate: Path, context: SensitivePathContext | None = None
) -> bool:
    """Whether ``candidate`` is a credential, gate-state, or app-database path.

    Comparison is by RESOLVED ancestry, never by string prefix, so
    ``~/.sshfoo`` is not mistaken for ``~/.ssh`` and a symlink cannot
    smuggle a path past the check. Each enumerated database's WAL/SHM/
    rollback-journal sidecar files are refused by the same exact-equality
    rule (see ``_db_sidecar_paths``), since they carry the same class of
    recent data as the database itself.

    This function only decides the question; it enforces nothing by
    itself. Callers -- ``ReadFileTool.execute``, ``WriteFileTool.execute``,
    ``ListDirectoryTool.execute``, ``GlobFiles.execute`` and
    ``GrepFiles.execute`` in ``Tools/file_operation_tools.py`` -- must call
    it (directly, or via that module's ``is_within``) explicitly on their
    target before touching the filesystem.

    Args:
        candidate: The path a tool intends to touch.
        context: An optional pre-resolved ``SensitivePathContext`` from
            ``resolve_sensitive_context()``. Pass one in when checking many
            candidates within a single tool invocation, so the sensitive-path
            set is resolved once instead of once per candidate. Leave this
            ``None`` (the default) for a one-off, single-path check -- that
            keeps this function's resolution genuinely per-call, which is
            what lets it observe a config-path switch (e.g. the test suite's
            ``TLDW_CONFIG_PATH`` swaps) without going stale.

    Returns:
        True when the path is refused. Fails CLOSED: a path that cannot be
        resolved is treated as sensitive.
    """
    resolved = _resolved(str(candidate))
    if resolved is None:
        return True

    ctx = context if context is not None else resolve_sensitive_context()

    for target in ctx.files:
        if resolved == target:
            return True

    for db_path in ctx.db_paths:
        if resolved == db_path:
            return True
        if resolved in _db_sidecar_paths(db_path):
            return True

    for root in ctx.dirs:
        if resolved == root or root in resolved.parents:
            return True

    return False
