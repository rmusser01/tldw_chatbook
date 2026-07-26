# tldw_chatbook/Utils/sensitive_paths.py
"""Paths refused by the agent-facing file tools, regardless of configured root.

Enforced directly inside ``Tools/file_operation_tools.py``'s ``ReadFileTool``,
``WriteFileTool`` and ``ListDirectoryTool`` -- each calls
:func:`is_sensitive_path` on its target immediately after ``validate_path``
and before touching the filesystem. It is *not* wired into
``Utils/path_validation.validate_path`` itself: that helper is the app's
general-purpose validator, used by first-party code (config screens, DB
path resolution, exports, ...) to validate paths to this application's own
config and database files, which are exactly the paths this module refuses.
Baking the check in there would block legitimate first-party access; it
belongs at the agent-tool boundary instead, shared by the `files` pack and
(from Phase 4) ``run_command``, so the two cannot drift.

Two distinct reasons a path lands here:

1. **Credentials.** ``read_file`` carries no risk tag, so it resolves to
   the built-in ``allow`` floor and executes with no prompt. An unconfined
   read is therefore a zero-prompt path from a private key into a
   persisted transcript that may be sent to any provider.
2. **This application's own gate state and data.** A tool able to rewrite
   ``mcp_permissions.json`` or ``config.toml`` can turn every ``ask`` into
   ``allow`` -- a one-step bypass of the permission system. A tool able to
   read or rewrite this app's own SQLite databases can exfiltrate or
   corrupt every conversation, note and credential-adjacent record they
   hold, bypassing the application layer entirely.

This is a guardrail, not a security boundary: it stops accidents and naive
injected payloads, not a determined ``python -c``. The sandbox track is
the real answer for shell execution.
"""

from __future__ import annotations

from pathlib import Path

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


def is_sensitive_path(candidate: Path) -> bool:
    """Whether ``candidate`` is a credential, gate-state, or app-database path.

    Comparison is by RESOLVED ancestry, never by string prefix, so
    ``~/.sshfoo`` is not mistaken for ``~/.ssh`` and a symlink cannot
    smuggle a path past the check.

    This function only decides the question; it enforces nothing by
    itself. Callers -- currently ``ReadFileTool.execute``,
    ``WriteFileTool.execute`` and ``ListDirectoryTool.execute`` in
    ``Tools/file_operation_tools.py`` -- must call it explicitly on their
    target before touching the filesystem.

    Args:
        candidate: The path a tool intends to touch.

    Returns:
        True when the path is refused. Fails CLOSED: a path that cannot be
        resolved is treated as sensitive.
    """
    resolved = _resolved(str(candidate))
    if resolved is None:
        return True

    for entry in _SENSITIVE_FILES:
        target = _resolved(entry)
        if target is not None and resolved == target:
            return True

    for db_path in _sensitive_db_paths():
        target = _resolved(str(db_path))
        if target is not None and resolved == target:
            return True

    for entry in _SENSITIVE_DIRS:
        root = _resolved(entry)
        if root is not None and (resolved == root or root in resolved.parents):
            return True

    return False
