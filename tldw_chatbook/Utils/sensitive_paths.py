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
refused, as a RULE rather than an enumeration (see the direct-child-file
loop in ``is_sensitive_path``): new state files land there constantly
(agent-run logs, eval/RAG-indexing/search-history/event/kanban/sync-state
DBs, ...) without ever touching ``config.py``, so an accessor-name
enumeration permanently trails reality. The SAME rule is applied to three
more directories, for the same reason: the effective config directory
(``config._get_effective_config_path().parent``, which honors
``TLDW_CONFIG_PATH`` the same way the config file itself does -- it holds
``config.toml``'s own ``.bak``/``.tmp`` backup sidecars plus
``runtime_policy.json``/``ui_state.toml``, none of which is enumerated by
name here either); the ChromaDB vector-store persist directory
(``RAG_Search.simplified.config.default_chroma_persist_directory()``,
which holds ``chroma.sqlite3`` -- plaintext chunks of the same
conversations and notes ``ChaChaNotes.db`` protects); and the RAG-profile
store (``RAG_Search.config_profiles.default_rag_profiles_dir()``, plaintext
per-profile RAG/embedding-provider config). Existing DIRECTORIES nested
directly under any of these four are excluded from the rule and stay fully
reachable -- most importantly the default file-tool sandbox root,
``get_user_data_dir() / "tool_sandbox"``; see that check's own comment for
why a directory/file distinction, not a name, is what exempts them.

The skill trust/grant store gets a DIFFERENT treatment: the WHOLE
``get_user_data_dir() / "skills" / "trust"`` subtree is refused, not just
its direct children, because ``skills`` itself is one of the exempted
container directories above and everything nested under it would otherwise
inherit that exemption -- see ``_sensitive_skill_trust_dir`` for why that
one subtree needs an explicit carve-out.

A directory can also be CREATED to collide with a not-yet-existing state
file at one of these locations (e.g. an agent asking ``write_file`` to
create parent directories for ``search_history.db/note.txt`` before this
app has ever created ``search_history.db`` as a file) -- the app's later
attempt to open its own state file then fails outright, a denial of
service. ``refuses_new_directory_chain`` is the guard against that: callers
that create directories on the agent's behalf (``WriteFileTool``'s
``create_directories=True`` path) must consult it before calling
``Path.mkdir(parents=True, ...)``.

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
    """Resolve a path string, returning ``None`` on ANY resolution failure.

    ``is_sensitive_path``'s fail-closed guarantee ("a path that cannot be
    resolved is treated as sensitive") depends on this returning ``None``
    for every way resolution can fail, not just the two most common ones.
    ``Path.resolve()``/``expanduser()`` normally raise ``OSError`` (e.g. a
    symlink loop) or ``RuntimeError`` (older Pythons' own loop-detection),
    but a path containing an embedded NUL byte raises ``ValueError``
    instead -- narrowing this catch to ``(OSError, RuntimeError)`` let that
    case escape ``is_sensitive_path`` entirely as an uncaught exception
    rather than the promised ``True`` (TASK-847). Broad by design: whatever
    exception ``pathlib`` raises for a candidate this function cannot make
    sense of, the caller must still get ``None`` back, never a propagated
    error.
    """
    try:
        return Path(path_str).expanduser().resolve()
    except Exception as exc:  # noqa: BLE001 - fail-closed for ANY resolution failure
        logger.debug(f"sensitive_paths: could not resolve {path_str!r}: {exc}")
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


def _sensitive_skill_trust_dir() -> Path | None:
    """Resolve this app's skill trust/grant store directory, lazily.

    ``get_user_data_dir() / "skills"`` is one of the existing-directory
    exemptions the direct-child-file rule (applied in ``is_sensitive_path``)
    carves out -- every trusted skill bundle lives as a named subdirectory
    under it, so a file inside it is deliberately NOT covered by that rule,
    letting agent tools browse/read a user's own skill bundles.

    The ``trust`` subdirectory nested one level inside it is the ONE
    exception carved back OUT of that exemption: it holds
    ``skill_trust_manifest.json`` (the authenticated trust manifest),
    ``skill_script_grants.json`` (the plain, UNAUTHENTICATED JSON file
    ``SkillTrustService.has_script_grant`` consults to authorize script
    EXECUTION -- deliberately kept outside the manifest's own HMAC+keyring
    integrity check; see ``Skills_Interop/skill_trust_service.py``),
    ``generation_marker.json`` (the local rollback-protection marker), and
    ``snapshots/`` (encrypted trusted-skill snapshots). A tool able to
    rewrite the grants file can authorize its own future script execution
    -- the same class of one-step gate bypass the MCP permission store's
    entry exists to prevent (see this module's docstring) -- so this
    caller refuses the WHOLE subtree by ancestry (the same way
    ``_SENSITIVE_DIRS`` is matched), not just its direct children: a file
    several levels inside ``snapshots/`` must be refused exactly like the
    manifest itself.

    Resolved via ``Skills_Interop.local_skills_service.default_local_skills_store_dir``
    and ``Skills_Interop.skill_trust_store.default_trust_store_dir`` -- the
    SAME functions ``app.py`` calls to build the live ``SkillTrustStore`` --
    never a re-spelled ``"skills"``/``"trust"`` literal, which would drift
    the moment either name changed (see this module's docstring for why
    that class of drift is exactly how a past finding went stale).

    Returns:
        The trust store directory, or ``None`` if ``get_user_data_dir()``
        could not be resolved.
    """
    from .. import config as _config
    from ..Skills_Interop.local_skills_service import default_local_skills_store_dir
    from ..Skills_Interop.skill_trust_store import default_trust_store_dir

    try:
        user_data_dir = _config.get_user_data_dir()
    except Exception as exc:  # noqa: BLE001 - defensive, additive coverage only
        logger.debug(f"sensitive_paths: could not resolve user data dir: {exc}")
        return None

    local_skills_store_dir = default_local_skills_store_dir(user_data_dir)
    return default_trust_store_dir(local_skills_store_dir)


def _direct_child_rule_container_dirs() -> tuple[Path, ...]:
    """Resolve every directory whose direct (non-recursive) child FILES are refused, lazily.

    Each of these is a directory this app treats as a bounded container for
    its own state, where new files land constantly without ever being
    named here individually -- this is the set the "Finding 2" rule in
    ``is_sensitive_path`` applies to. Existing DIRECTORIES nested directly
    inside any one of them (``tool_sandbox``, ``chat_dicts``, ``chromadb``,
    ``exports``, ``rag_profiles``, ``skills``, and any future sibling) are
    exempt from the rule and stay fully reachable; only a same-level FILE
    is refused. See that rule's own comment for why "is an existing
    directory", not a name, is what exempts them.

    Returns:
        Every container directory whose accessor could be resolved:

        * ``config.get_user_data_dir()``.
        * The effective config directory
          (``config._get_effective_config_path().parent``) -- honors
          ``TLDW_CONFIG_PATH`` the same way the config file itself does.
          This is what covers ``config.toml``'s own ``.bak``/``.tmp``
          backup sidecars (``UI/Screens/settings_screen.py``'s Advanced
          config save writes both, byte-identical to the live config,
          API keys included) and any other loose file dropped beside it
          (``runtime_policy.json``, ``ui_state.toml``, a hand-made backup
          copy under any other name) -- none of which is enumerated here
          by name either, for the same reason the user-data-dir rule
          isn't: an enumeration permanently trails whatever gets written
          there next.
        * The ChromaDB vector-store persist directory
          (``RAG_Search.simplified.config.default_chroma_persist_directory()``),
          which holds ``chroma.sqlite3`` -- plaintext chunks of the same
          conversations and notes ``ChaChaNotes.db`` protects.
        * The RAG-profile store directory
          (``RAG_Search.config_profiles.default_rag_profiles_dir()``),
          plaintext per-profile RAG/embedding-provider config.

        An accessor that raises is skipped rather than failing the whole
        check, as elsewhere in this module.
    """
    from .. import config as _config
    from ..RAG_Search.config_profiles import default_rag_profiles_dir
    from ..RAG_Search.simplified.config import default_chroma_persist_directory

    resolved: list[Path] = []

    try:
        resolved.append(_config.get_user_data_dir())
    except Exception as exc:  # noqa: BLE001 - defensive, additive coverage only
        logger.debug(f"sensitive_paths: could not resolve user data dir: {exc}")

    try:
        resolved.append(_config._get_effective_config_path().parent)
    except Exception as exc:  # noqa: BLE001 - defensive, additive coverage only
        logger.debug(f"sensitive_paths: could not resolve effective config dir: {exc}")

    try:
        resolved.append(default_chroma_persist_directory())
    except Exception as exc:  # noqa: BLE001 - defensive, additive coverage only
        logger.debug(f"sensitive_paths: could not resolve chroma persist dir: {exc}")

    try:
        resolved.append(default_rag_profiles_dir())
    except Exception as exc:  # noqa: BLE001 - defensive, additive coverage only
        logger.debug(f"sensitive_paths: could not resolve rag profiles dir: {exc}")

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
    #: be resolved. Kept as its own field for callers/tests that care about
    #: this one specific directory; the direct-child-file rule itself now
    #: consults ``direct_child_denied_dirs`` below, which already includes
    #: this value alongside the other container directories that get the
    #: same treatment.
    user_data_dir: Path | None
    #: Every directory whose direct (non-recursive) child FILES are
    #: refused -- ``user_data_dir``, the effective config directory, the
    #: ChromaDB persist directory, and the RAG-profile store directory (see
    #: ``_direct_child_rule_container_dirs``). Entries that failed to
    #: resolve are dropped, same as ``files``/``dirs``/``db_paths``.
    direct_child_denied_dirs: tuple[Path, ...]


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

    skill_trust_dir = _sensitive_skill_trust_dir()
    dynamic_dirs = (skill_trust_dir,) if skill_trust_dir is not None else ()

    return SensitivePathContext(
        files=tuple(
            p
            for p in (_resolved(str(raw)) for raw in _sensitive_single_file_paths())
            if p is not None
        ),
        dirs=tuple(
            p
            for p in (_resolved(str(entry)) for entry in _SENSITIVE_DIRS + dynamic_dirs)
            if p is not None
        ),
        db_paths=tuple(
            p
            for p in (_resolved(str(raw)) for raw in _sensitive_db_paths())
            if p is not None
        ),
        user_data_dir=user_data_dir,
        direct_child_denied_dirs=tuple(
            p
            for p in (
                _resolved(str(raw)) for raw in _direct_child_rule_container_dirs()
            )
            if p is not None
        ),
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

    # Finding 2 (substrate review), generalized beyond `get_user_data_dir()`
    # to every container directory `_direct_child_rule_container_dirs()`
    # resolves (also the effective config directory, the ChromaDB persist
    # directory, and the RAG-profile store -- TASK-848): refuse every FILE
    # sitting directly (non-recursively) inside one of them, as a RULE
    # rather than an enumeration. New state files land there constantly
    # without ever touching config.py -- agent-run logs, eval/RAG-indexing/
    # search-history/event/kanban/sync-state DBs, the MCP local-store/
    # context JSON files, the rotating app log, config.toml's own
    # `.bak`/`.tmp` backup sidecars -- and an accessor-name enumeration
    # (`_DB_PATH_ACCESSOR_NAMES` above) permanently trails whatever the app
    # actually creates there next.
    #
    # Checked by "is it a directory", never by name: every legitimate use
    # of one of these directories as a CONTAINER creates a named
    # subdirectory instead of a loose file directly inside it -- e.g.
    # `tool_sandbox` (the default file-tool sandbox root itself),
    # `chat_dicts`, `chromadb`, `exports`, `rag_profiles`, `skills` nested
    # under `get_user_data_dir()`. Excluding "is an existing directory"
    # rather than hardcoding any of those names keeps every one of them
    # reachable, including ones added later, without needing this rule to
    # be updated in lockstep -- while a candidate that does not exist yet
    # (e.g. a `write_file` target for a brand-new file) is NOT a directory
    # either, so it still fails closed and is refused. TASK-849: that same
    # gate means an agent COULD plant a directory at a name the app has
    # never used yet (before this check ever sees it as "existing") --
    # closing that hole is `refuses_new_directory_chain` below, consulted
    # by callers BEFORE they create a directory on the agent's behalf,
    # never by loosening this check's own "is a directory" gate (which
    # would break every legitimate container above).
    for denied_parent in ctx.direct_child_denied_dirs:
        if resolved.parent == denied_parent and not resolved.is_dir():
            return True

    return False


def refuses_new_directory_chain(
    target_dir: Path, context: SensitivePathContext | None = None
) -> bool:
    """Whether creating ``target_dir`` (or any not-yet-existing parent of it)
    would plant a directory where this app expects a plain state file.

    ``is_sensitive_path``'s direct-child-file rule is deliberately gated on
    "does this candidate already exist as a directory", so a pre-existing
    container (``tool_sandbox``, ``chromadb``, ``skills``, ...) stays fully
    reachable. That same gate means a candidate that does NOT yet exist is
    judged as if it were a plain file -- correctly refused. But
    ``WriteFileTool``'s ``create_directories=True`` path only ever validates
    the FINAL file being written, never the new directory levels
    ``Path.mkdir(parents=True)`` creates on the way there: a target like
    ``search_history.db/note.txt`` has a parent (``.../search_history.db``)
    that is never itself checked, so nothing stopped an agent from planting
    a directory at that exact name before this app ever created
    ``search_history.db`` as a SQLite file (TASK-849, verified reachable
    end to end through ``WriteFileTool`` under a widened sandbox root). The
    app's own later ``sqlite3.connect(...)`` (or equivalent open) then fails
    outright -- a denial of service, not a disclosure: the collision itself
    carries no credential and grants no elevated access.

    Walking upward from ``target_dir`` while each level still does not
    exist mirrors exactly what ``Path.mkdir(parents=True)`` is about to
    create, and checks each such level with ``is_sensitive_path`` --
    reusing the exact same direct-child-file rule, never a separate check.
    Any level found to already exist ends the walk immediately: an existing
    ancestor is never touched by ``mkdir(parents=True)``, so nothing new
    needs checking above it -- which is what keeps every legitimate
    container directory (created by the app itself before an agent tool
    ever runs) fully reachable.

    Consequence worth naming (Finding 2, follow-up hardening review): a
    not-yet-existing name always fails ``is_sensitive_path``'s ``is_dir()``
    gate, so this refuses creating **any** brand-new subdirectory directly
    inside one of the container directories the direct-child-file rule
    protects (``get_user_data_dir()``, the ChromaDB persist directory,
    ...) -- not only a name that happens to collide with a state file this
    app actually uses. Reproduced: with the sandbox root widened to
    contain the ChromaDB persist directory,
    ``write_file("chromadb/newcoll/x.txt", create_directories=True)`` is
    refused (``newcoll`` does not exist yet, so it fails the same gate a
    genuine collision would), while ``write_file("chromadb/coll1/new.txt",
    create_directories=True)`` succeeds once ``coll1`` already exists as a
    directory -- the walk stops at the first already-existing ancestor, as
    documented above. This is deliberate, not a bug to fix here: telling
    "a legitimate brand-new container" apart from "a shadow directory
    aimed at a not-yet-created state file" by name alone would require
    exactly the enumeration this design avoids (see the module
    docstring), so failing closed is the right default. It is only
    reachable when the sandbox root (or a bound workspace folder) is
    widened to actually contain one of these container directories -- the
    default sandbox root never does. Noted here so the next reader is not
    surprised by an agent's brand-new-subdirectory `write_file` call being
    refused under such a configuration.

    Args:
        target_dir: The directory ``mkdir(parents=True)`` is about to
            create -- typically a write target's parent directory.
        context: Optional pre-resolved ``SensitivePathContext``; see
            ``resolve_sensitive_context``.

    Returns:
        True if ``target_dir`` or any of its not-yet-existing ancestors
        would be a sensitive path once created.
    """
    ctx = context if context is not None else resolve_sensitive_context()
    node = target_dir
    while True:
        resolved = _resolved(str(node))
        if resolved is None:
            return True
        if resolved.exists():
            return False
        if is_sensitive_path(resolved, context=ctx):
            return True
        parent = node.parent
        if parent == node:
            return False
        node = parent
