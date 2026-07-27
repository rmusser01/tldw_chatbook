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

Every one of those is resolved through the app's OWN accessors at call
time, never a hardcoded literal: ``config.toml``'s location honors the
``TLDW_CONFIG_PATH`` override (``config._get_effective_config_path()``),
the MCP permission store and its companions live under
``config.get_user_data_dir()`` (never under the ``~/.config/tldw_cli/``
literal a first look at ``app.py`` might suggest -- see
``_sensitive_single_file_paths()``), and the SQLite DB paths honor
``[database]`` overrides and the active user folder (see
``_sensitive_db_paths()``). A literal here would drift the moment any of
those is overridden -- which is exactly how the permission-store literal
went stale (Finding 1) and how a ``TLDW_CONFIG_PATH`` override defeated the
``config.toml`` entry (Finding 3).

Every file this app creates directly under ``get_user_data_dir()`` is also
refused, as a RULE rather than an enumeration (see the
``resolved.parent == ctx.user_data_dir`` check in ``is_sensitive_path``):
new state files land there constantly (agent-run logs, eval/RAG-indexing/
search-history/event/kanban/sync-state DBs, ...) without ever touching
``config.py``, so an accessor-name enumeration permanently trails reality.
Existing DIRECTORIES nested there -- most importantly the default file-tool
sandbox root, ``get_user_data_dir() / "tool_sandbox"`` -- are excluded from
that rule and stay fully reachable; see that check's own comment for why a
directory/file distinction, not a name, is what exempts them.

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

#: NOTE: this app's own ``config.toml`` and the MCP-permission-store family
#: used to be listed here as static literals (``~/.config/tldw_cli/
#: config.toml``, ``~/.config/tldw_cli/mcp_permissions.json``). Both can
#: move at runtime -- ``config.toml`` honors the ``TLDW_CONFIG_PATH``
#: override, and the permission store's REAL location was never actually
#: ``~/.config/tldw_cli/`` at all; the app builds it under
#: ``get_user_data_dir()`` (see ``_sensitive_single_file_paths()`` below for
#: exactly how). A literal here would silently stop matching the moment
#: either moved -- which is precisely how the permission-store entry went
#: stale (Finding 1) and the ``config.toml`` entry missed a
#: ``TLDW_CONFIG_PATH`` override (Finding 3). Both are now resolved lazily,
#: the same way the DB paths are, by ``_sensitive_single_file_paths()``.

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
    not beneath it, so the static ``_SENSITIVE_DIRS`` tuple above cannot
    express their location. Each path is resolved via the app's own
    accessor (which also honors ``[database]`` path overrides and the
    active user folder) rather than hardcoded, since neither the user
    folder nor an override is known statically.

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


def _sensitive_single_file_paths() -> tuple[Path, ...]:
    """Resolve this app's own non-DB sensitive single files, lazily.

    Two families, each resolved through the same accessor the app itself
    uses to build the real path -- never a literal -- because both can move
    at runtime:

    1. **config.toml.** ``config._get_effective_config_path()`` honors the
       ``TLDW_CONFIG_PATH`` override (set throughout this project's own
       test suite, and by any deployment that relocates the config file).
       A literal default-path check misses the file actually holding the
       user's API keys whenever that override is set (Finding 3).
    2. **The MCP permission store and its companions.** The store's real
       path is ``get_user_data_dir() / "mcp_permissions.json"`` -- built by
       ``MCP.unified_control_plane_service``'s ``permission_store`` property
       as ``Path(store.path).with_name("mcp_permissions.json")``, where
       ``store.path`` is the ``LocalMCPStore`` path ``app.py`` constructs as
       ``get_user_data_dir() / "local_mcp_store.json"``. A tool able to
       rewrite this file can turn every ``ask`` into ``allow`` -- the
       CRITICAL one-step permission-gate bypass this module exists to
       prevent (Finding 1; see the module docstring). Two companions built
       the exact same ``Path(...).with_name(...)`` way from that same base
       path carry the same class of gate-relevant state:
       ``local_mcp_store.json`` itself (server definitions and their env)
       and ``mcp_execution_log.jsonl`` (the execution audit trail).

    Returns:
        Resolved paths for every file above whose accessor could be
        called. An accessor that raises is skipped rather than failing the
        whole check -- additional coverage, not the primary guarantee (see
        ``_sensitive_db_paths``, which does the same for the DB paths).
    """
    from .. import config as _config

    resolved: list[Path] = []

    try:
        resolved.append(_config._get_effective_config_path())
    except Exception as exc:  # noqa: BLE001 - defensive, additive coverage only
        logger.debug(f"sensitive_paths: could not resolve config.toml path: {exc}")

    try:
        user_data_dir = _config.get_user_data_dir()
    except Exception as exc:  # noqa: BLE001 - defensive, additive coverage only
        logger.debug(f"sensitive_paths: could not resolve user data dir: {exc}")
    else:
        resolved.append(user_data_dir / "mcp_permissions.json")
        resolved.append(user_data_dir / "local_mcp_store.json")
        resolved.append(user_data_dir / "mcp_execution_log.jsonl")

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
    #: Resolved ``config.get_user_data_dir()``, or ``None`` if it could not
    #: be resolved. Backs the Finding-2 rule in ``is_sensitive_path``: every
    #: FILE sitting directly (non-recursively) inside this directory is
    #: refused, regardless of whether it is one of the enumerated DBs above.
    #: ``None`` simply means that rule does not fire for this context --
    #: ``files``/``dirs``/``db_paths`` coverage is unaffected either way.
    user_data_dir: Path | None


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
        sensitive files, directories, database paths, and user data
        directory (entries that failed to resolve are dropped; the user
        data directory is ``None`` if it could not be resolved).
    """
    from .. import config as _config

    try:
        user_data_dir = _resolved(str(_config.get_user_data_dir()))
    except Exception as exc:  # noqa: BLE001 - defensive, additive coverage only
        logger.debug(f"sensitive_paths: could not resolve user data dir: {exc}")
        user_data_dir = None

    return SensitivePathContext(
        files=tuple(
            p
            for p in (_resolved(str(raw)) for raw in _sensitive_single_file_paths())
            if p is not None
        ),
        dirs=tuple(
            p for p in (_resolved(entry) for entry in _SENSITIVE_DIRS) if p is not None
        ),
        db_paths=tuple(
            p
            for p in (_resolved(str(raw)) for raw in _sensitive_db_paths())
            if p is not None
        ),
        user_data_dir=user_data_dir,
    )


def is_sensitive_path(
    candidate: Path, context: SensitivePathContext | None = None
) -> bool:
    """Whether ``candidate`` is a credential, gate-state, or app-state path.

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

    # Finding 2 (substrate review): refuse every FILE sitting directly
    # (non-recursively) inside `get_user_data_dir()`, as a RULE rather than
    # an enumeration. New state files land there constantly without ever
    # touching config.py -- agent-run logs, eval/RAG-indexing/search-
    # history/event/kanban/sync-state DBs, the MCP local-store/context JSON
    # files, the rotating app log -- and an accessor-name enumeration
    # (`_DB_PATH_ACCESSOR_NAMES` above) permanently trails whatever the app
    # actually creates there next.
    #
    # Checked by "is it a directory", never by name: every legitimate use
    # of this directory as a CONTAINER creates a named subdirectory instead
    # of a loose file directly inside it -- `tool_sandbox` (the default
    # file-tool sandbox root itself), `chat_dicts`, `chromadb`, `exports`,
    # `rag_profiles`, `skills`. Excluding "is an existing directory" rather
    # than hardcoding any of those names keeps every one of them reachable,
    # including ones added later, without needing this rule to be updated
    # in lockstep -- while a candidate that does not exist yet (e.g. a
    # `write_file` target for a brand-new file) is NOT a directory either,
    # so it still fails closed and is refused.
    if ctx.user_data_dir is not None and resolved.parent == ctx.user_data_dir:
        if not resolved.is_dir():
            return True

    return False
