"""Segmented, append-only run log writer.

Impure by design (filesystem + config). Path resolution reuses the file
tools' own chain -- ``allowed_file_roots`` -> ``is_within`` ->
``is_sensitive_path`` -- so this writer can never become a path-validation
bypass. See the design spec §3.3, §7, §8, §9.2.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from loguru import logger

from .run_log_format import RunLogRecord, encode_record

#: F3 (Qodo #3): CLAUDE.md mandates "env vars -> config.toml -> defaults",
#: but `_setting` below previously never consulted the environment at all.
#: Named ``TLDW_AGENTS_<KEY>`` to match this repo's existing per-setting
#: override convention -- e.g. ``TLDW_CONSOLE_LLAMA_CPP_BASE_URL`` in
#: UI/Screens/chat_screen.py is ``TLDW_`` + the config SECTION
#: (``console``) + the key. The ``[agents]`` section name gives the middle
#: segment here.
_ENV_PREFIX = "TLDW_AGENTS_"
_ENV_TRUE = {"1", "true", "yes", "on"}
_ENV_FALSE = {"0", "false", "no", "off"}

#: Directory created inside the resolved root, in BOTH the sandbox-fallback
#: and the bound-workspace case: `bind()` always dots this name
#: (`.agent-runs`) before creating it. `_is_hidden_within`
#: (`Tools/file_operation_tools.py`) excludes any dot-prefixed path
#: component from `glob_files`/`grep_files`/`read_file`, so a sub-agent --
#: which inherits its parent's tool allow-list -- can never reach the
#: parent's run log through those tools, regardless of which root (the
#: sandbox, or a bound workspace folder) the log happened to land under.
#:
#: History (TASK-1270, 2026-07-28): this was originally a conditional --
#: dotted only under the sandbox fallback, undotted for a bound workspace
#: folder. The sandbox-only dotting was a real, reviewed fix for a
#: sub-agent log-disclosure bug (a child could `grep_files` its parent's
#: log and extract secrets). Staying undotted for a bound workspace folder
#: was CORRECT when that decision was made: at the time,
#: `glob_files`/`grep_files` globbed `_tool_sandbox_root()` alone and could
#: not reach a workspace folder root at all, so undotted there cost
#: nothing and bought the log user-visibility inside the user's own
#: project. TASK-850 ("Scope glob_files and grep_files to workspace folder
#: roots") invalidated that premise: both tools now resolve every root
#: `allowed_file_roots()` returns, so an undotted workspace-folder log
#: became reachable by a sub-agent through them -- the exact disclosure
#: the sandbox-fallback dotting was meant to prevent, reopened for the
#: workspace case by an unrelated change. TASK-1270 reproduced this with a
#: planted secret (`Tests/Agents/test_run_log_workspace_isolation.py`) and
#: closed it by dotting the name unconditionally, deleting the
#: sandbox-vs-workspace conditional (and the `_root_kind` side channel it
#: depended on) entirely: the security property must not depend on which
#: root was chosen, nor on what the generic file tools happen to search
#: this month -- a uniform rule cannot rot the way that conditional did.
#:
#: What this does NOT give up: a dotted directory is still an ORDINARY
#: directory to the user -- `ls -a` lists it, editors show it, it is fully
#: diffable and keepable in the user's own repository. It is hidden only
#: from this app's own sandboxed file tools, which is precisely the
#: point. `search_run_log`'s own reader (`run_log_search.load_records`) is
#: unaffected either way: it globs `log_dir` directly and never routes
#: through `validate_path`/`_is_hidden_within`, which is what rejects a
#: hidden path component.
DEFAULT_DIR_NAME = "agent-runs"
DEFAULT_SEGMENT_BYTES = 4_000_000
DEFAULT_MAX_RECORD_BYTES = 1_000_000
MANIFEST_NAME = "MANIFEST"


def _env_override(key: str) -> str | None:
    """Read ``TLDW_AGENTS_<KEY>`` from the environment, or ``None`` if unset.

    F3 (Qodo #3): the highest-priority tier below an explicit constructor
    argument (CLAUDE.md: "env vars -> config.toml -> defaults"). Only a
    non-empty string counts as "set" -- an env var present but blank falls
    through to the TOML/default tiers, same as an unset one.

    Args:
        key: Key name within the ``[agents]`` section (e.g. ``run_log_dir_name``).

    Returns:
        The raw environment string, or ``None`` when not set to a
        non-empty value.
    """
    value = os.environ.get(f"{_ENV_PREFIX}{key.upper()}")
    return value if value not in (None, "") else None


def _parse_env_bool(raw: str, key: str, default: bool) -> bool:
    """Parse an env-var override for a boolean ``[agents]`` setting.

    Args:
        raw: The raw environment string (already confirmed non-empty).
        key: Setting key, named in the warning on an unrecognised value.
        default: Value returned when ``raw`` cannot be parsed as a boolean.

    Returns:
        ``True``/``False`` for a recognised token (case-insensitive
        ``1``/``true``/``yes``/``on`` or ``0``/``false``/``no``/``off``),
        else ``default``.
    """
    lowered = raw.strip().lower()
    if lowered in _ENV_TRUE:
        return True
    if lowered in _ENV_FALSE:
        return False
    logger.warning(
        f"run log: TLDW_AGENTS_{key.upper()}={raw!r} is not a recognised "
        f"boolean; using default"
    )
    return default


def _setting(key: str, default):
    """Read one ``[agents]`` config key: env var, then TOML, then default.

    F3 (Qodo #3): CLAUDE.md's documented priority is "env vars ->
    config.toml -> defaults", but this previously skipped the env tier
    entirely. An explicit constructor argument on ``RunLogWriter`` (see
    ``__init__``) short-circuits this function altogether and is therefore
    still the highest-priority override overall -- this only fixes the
    ordering of the two tiers BELOW that.

    Args:
        key: Key name within the ``[agents]`` section.
        default: Value returned when unset or unreadable.

    Returns:
        The env override (boolean-parsed when ``default`` is a ``bool``,
        else the raw string) when set; otherwise the configured TOML value;
        otherwise ``default``.
    """
    env_value = _env_override(key)
    if env_value is not None:
        if isinstance(default, bool):
            return _parse_env_bool(env_value, key, default)
        return env_value
    try:
        from tldw_chatbook.config import get_cli_setting

        value = get_cli_setting("agents", key, default)
    except Exception:
        return default
    return default if value is None else value


def _coerce_dir_name(value, default: str) -> str:
    """Coerce dir_name defensively into a single, safe path COMPONENT.

    F1 (Qodo #1, PR #1066 review ruling): ``dir_name`` is configurable
    (``[agents] run_log_dir_name`` or an explicit constructor argument) and
    therefore untrusted like any config value, and it is joined directly
    onto the resolved log root (``root / dir_name`` in ``bind()``) --
    pathlib's ``/`` operator REPLACES the whole path outright when the
    right-hand side is absolute, so an unvalidated value could silently
    redirect the log tree entirely rather than merely fail the later
    containment check. This rejects a separator, ``.``/``..``, an absolute
    form, or an empty/whitespace-only value UP FRONT and falls back to
    ``default`` (logged at warning) so a bad config value degrades to
    "log under the default name" rather than "logging silently disabled" --
    a config typo should not be able to kill a crash-durability feature.
    ``bind()``'s existing ``allowed_file_roots`` -> ``is_within``
    containment check on the ASSEMBLED path remains as defense in depth
    (it is what actually matters for ``run_id``, which is not vetted here).

    Deliberately NOT routed through ``Utils/path_validation.validate_path``:
    that function raises "Access to hidden files/directories is not
    allowed" on any hidden (dotted) path component, and ``bind()``
    intentionally dots this directory UNCONDITIONALLY (TASK-1270: both
    under the sandbox fallback and for a bound workspace folder) -- a
    real, reviewed fix for a sub-agent log-disclosure bug (a child could
    ``grep_files``/``glob_files`` its parent's log and extract secrets;
    see the module-level comment above ``DEFAULT_DIR_NAME``). Routing
    through ``validate_path`` would reject ``.agent-runs`` outright and
    disable logging in the DEFAULT configuration -- do not "fix" this by
    reaching for that function.

    Args:
        value: Value to coerce (from explicit arg or config).
        default: Default value if coercion fails.

    Returns:
        A safe, single-path-component directory name, or ``default``.
    """
    try:
        s = str(value).strip()
        if not s:
            raise ValueError("empty dir_name")
        if "/" in s or "\\" in s:
            raise ValueError(f"dir_name contains a path separator: {s!r}")
        if s in (".", ".."):
            raise ValueError(f"dir_name is a path-traversal segment: {s!r}")
        if Path(s).is_absolute():
            raise ValueError(f"dir_name is absolute: {s!r}")
    except Exception:
        logger.opt(exception=True).warning("run log: invalid dir_name, using default")
        return default
    return s


def _coerce_positive_int(value, default: int, name: str) -> int:
    """Coerce to positive int defensively: non-numeric, zero, or negative falls back to default.

    Args:
        value: Value to coerce (from explicit arg or config).
        default: Default value if coercion fails.
        name: Parameter name for logging (e.g., "segment_bytes").

    Returns:
        A positive integer, or ``default``.
    """
    try:
        val = int(value)
        if val <= 0:
            raise ValueError(f"non-positive {name}")
        return val
    except Exception:
        logger.opt(exception=True).warning(
            f"run log: invalid {name}, using default"
        )
        return default


def resolve_log_root() -> Path | None:
    """Return the directory the log tree is created under, or ``None``.

    Prefers the run's first read-write workspace folder root so the log is
    a user-visible artifact; falls back to the tool sandbox root when no
    such folder is bound. Any failure resolves to ``None`` (logging off)
    rather than to a wider or unvalidated location.

    TASK-1270: this used to also report, via a thread-local side channel,
    whether the call resolved via the sandbox fallback -- ``bind()`` read
    that back to decide whether to dot the log directory name. Since
    ``bind()`` now dots the name unconditionally regardless of which root
    this function chose (see the comment above ``DEFAULT_DIR_NAME``), that
    side channel had no remaining purpose and was removed. The return
    value is a bare ``Path | None``, exactly as before.

    Returns:
        The chosen root directory, or ``None`` when none is usable.
    """
    try:
        from tldw_chatbook.Tools.file_operation_tools import _tool_sandbox_root
        from tldw_chatbook.Tools.workspace_file_roots import allowed_file_roots

        sandbox = _tool_sandbox_root()
        roots = allowed_file_roots(write=True, sandbox_root=sandbox)
    except Exception:
        logger.opt(exception=True).warning("run log: cannot resolve any root")
        return None
    if not roots:
        return None
    # allowed_file_roots returns (sandbox, *workspace_folders); prefer a
    # bound workspace folder, fall back to the sandbox. Which branch fires
    # no longer affects the log directory's NAME (TASK-1270) -- only which
    # root it is created under.
    for candidate in roots[1:]:
        return candidate
    return roots[0]


class RunLogRecordNumber(int):
    """A record number that also reports whether the LOG capped this record.

    F7 (Qodo #7): a caller following a truncation trailer's pointer needs to
    know whether the pointed-at record is itself complete -- ``append()``
    already knows this (it is the same comparison that decides whether to
    set ``truncated_from`` on the record), but plumbing it back to
    ``agent_runtime._truncate_tool_result`` without this class would mean
    either changing ``append()``'s / ``LoopDeps.on_record``'s return shape
    (which ``test_on_record_returns_the_assigned_record_number`` pins as a
    plain ``int``) or adding a whole new ``LoopDeps`` callable for one
    boolean. Subclassing ``int`` instead means every existing caller --
    equality, comparison, ``%06d`` formatting, ``isinstance(x, int)``,
    hashing/set membership -- sees no difference at all, while a caller that
    knows to look can read ``.truncated``. Callers that don't know about
    this class read it exactly like a plain ``int`` and are unaffected.
    """

    truncated: bool

    def __new__(cls, value: int, *, truncated: bool) -> "RunLogRecordNumber":
        obj = super().__new__(cls, value)
        obj.truncated = truncated
        return obj


class RunLogWriter:
    """Appends records for ONE run tree to a segmented log.

    Constructed unbound (the run id does not exist until ``_run_one`` calls
    ``create_run``), then bound once by the primary run. Child runs share
    the instance, and therefore the record counter, so parent and child
    record numbers can never collide.
    """

    def __init__(
        self,
        *,
        dir_name: str | None = None,
        segment_bytes: int | None = None,
        max_record_bytes: int | None = None,
    ) -> None:
        """Build an UNBOUND writer. Explicit args override ``[agents]`` config.

        Args:
            dir_name: Directory name; defaults to ``[agents] run_log_dir_name``.
            segment_bytes: Roll threshold; defaults to
                ``[agents] run_log_segment_bytes``.
            max_record_bytes: Per-record ceiling; defaults to
                ``[agents] run_log_max_record_bytes``.
        """
        # Coerce dir_name defensively using shared helper.
        if dir_name is not None:
            self._dir_name = _coerce_dir_name(dir_name, DEFAULT_DIR_NAME)
        else:
            configured = _setting("run_log_dir_name", DEFAULT_DIR_NAME)
            self._dir_name = _coerce_dir_name(configured, DEFAULT_DIR_NAME)

        # Coerce segment_bytes defensively using shared helper.
        if segment_bytes is not None:
            self._segment_bytes = _coerce_positive_int(
                segment_bytes, DEFAULT_SEGMENT_BYTES, "segment_bytes"
            )
        else:
            configured = _setting("run_log_segment_bytes", DEFAULT_SEGMENT_BYTES)
            self._segment_bytes = _coerce_positive_int(
                configured, DEFAULT_SEGMENT_BYTES, "segment_bytes"
            )

        # Coerce max_record_bytes defensively using shared helper.
        if max_record_bytes is not None:
            self._max_record_bytes = _coerce_positive_int(
                max_record_bytes, DEFAULT_MAX_RECORD_BYTES, "max_record_bytes"
            )
        else:
            configured = _setting("run_log_max_record_bytes", DEFAULT_MAX_RECORD_BYTES)
            self._max_record_bytes = _coerce_positive_int(
                configured, DEFAULT_MAX_RECORD_BYTES, "max_record_bytes"
            )

        self._lock = threading.Lock()
        self._counter = 0
        self._segment_index = 1
        self._segment_size = 0
        self._active = False
        self._bind_attempted = False  # Track whether bind() was called, success or failure
        self.log_dir: Path | None = None

    @property
    def is_active(self) -> bool:
        """Whether records are currently being written.

        Returns:
            ``True`` once ``bind()`` has successfully activated the writer
            (root resolved, directory created); ``False`` before ``bind()``
            is called, after a failed bind (disabled config, unresolvable
            root, or a directory-creation failure), or once a later write
            failure has deactivated it (see ``append()``).
        """
        return self._active

    def bind(self, run_id: str) -> None:
        """Bind to ``run_id`` and create its directory. Idempotent.

        Args:
            run_id: The PRIMARY run's id. Later calls are ignored so a
                child run never rebinds its parent's writer.
        """
        if self._bind_attempted:
            return
        self._bind_attempted = True

        if not _setting("run_log_enabled", True):
            self._active = False
            return
        root = resolve_log_root()
        if root is None:
            self._active = False
            return
        dir_name = self._dir_name
        if not dir_name.startswith("."):
            # TASK-1270: dotted UNCONDITIONALLY -- in both the
            # sandbox-fallback and the bound-workspace case. See the
            # module-level comment above `DEFAULT_DIR_NAME` for the full
            # history: this used to depend on WHICH root `resolve_log_root`
            # picked (a conditional that was correct when written and
            # silently stopped being true when TASK-850 changed
            # `glob_files`/`grep_files`'s reach). The security property
            # must not depend on which root was chosen, nor on what the
            # generic file tools happen to search this month -- a uniform
            # rule cannot rot the way that conditional did. Dotting here
            # does not break OUR OWN reader: `search_run_log` ->
            # `run_log_search.load_records` globs this directory directly
            # and never routes through `validate_path`/`_is_hidden_within`
            # -- it only removes the directory from what
            # `glob_files`/`grep_files`/`read_file` can see, which is
            # exactly the intent.
            dir_name = f".{dir_name}"
        try:
            from tldw_chatbook.Tools.file_operation_tools import is_within

            base = root / dir_name
            # Verify containment before creating any directories.
            if not is_within(base, root):
                logger.warning(
                    "run log: base directory escapes root; logging disabled"
                )
                self._active = False
                return
            base.mkdir(parents=True, exist_ok=True)
            gitignore = base / ".gitignore"
            if not gitignore.exists():
                # Created only if absent: writing into a user's repository
                # is itself a mutation.
                gitignore.write_text("*\n", encoding="utf-8")
            run_dir = base / run_id
            # Verify containment of run_dir before creating it.
            if not is_within(run_dir, root):
                logger.warning(
                    "run log: run directory escapes root; logging disabled"
                )
                self._active = False
                return
            run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.opt(exception=True).warning(
                "run log: cannot create log directory; logging disabled"
            )
            self._active = False
            return
        self.log_dir = run_dir
        self._active = True

    def _segment_path(self) -> Path:
        assert self.log_dir is not None
        return self.log_dir / f"logs.{self._segment_index:04d}.txt"

    def _write_bytes(self, path: Path, payload: bytes, *, sync: bool = False) -> None:
        """Append ``payload`` to ``path``.

        ``flush()`` on every record survives a process crash. ``fsync`` is
        reserved for segment rolls and run end (``sync=True``): calling it
        per record into a user's project directory is wasteful.

        Args:
            path: Target file, opened in append-binary mode.
            payload: Bytes to append.
            sync: Whether to force an ``fsync`` after flushing.
        """
        import os

        with open(path, "ab") as handle:
            handle.write(payload)
            handle.flush()
            if sync:
                os.fsync(handle.fileno())

    def append(
        self,
        *,
        run_id: str,
        kind: str,
        type: str,
        content: str,
        tool: str = "",
        status: str = "",
        call_id: str = "",
    ) -> int | None:
        """Append one record and return its number.

        Args:
            run_id: Id of the run this record belongs to (parent or child).
            kind: ``primary`` or ``subagent``.
            type: ``model``, ``tool_call``, ``tool_result``, or ``spawn``.
            content: Full, untruncated text.
            tool: Tool name, when applicable.
            status: ``ok`` / ``error``, when applicable.
            call_id: Provider ``tool_call_id``, when applicable.

        Returns:
            The assigned record number as a ``RunLogRecordNumber`` (a plain
            ``int`` for every purpose -- equality, formatting, hashing --
            plus a ``.truncated`` attribute reporting whether THIS record's
            content exceeded ``run_log_max_record_bytes`` and was cut), or
            ``None`` when the writer is inactive or the write failed. Never
            raises.
        """
        if not self._active or self.log_dir is None:
            return None
        with self._lock:
            truncated_from = 0
            body = content.encode("utf-8")
            if len(body) > self._max_record_bytes:
                truncated_from = len(body)
                # Cut on a character boundary, then re-encode.
                body = body[: self._max_record_bytes]
                content = body.decode("utf-8", "ignore")
            self._counter += 1
            record = RunLogRecord(
                number=self._counter,
                run_id=run_id,
                kind=kind,
                type=type,
                ts=_now_iso(),
                content=content,
                tool=tool,
                status=status,
                call_id=call_id,
                truncated_from=truncated_from,
            )
            payload = encode_record(record)
            # Roll BEFORE writing: a record must never span segments, or
            # bytes=-exact parsing (which assumes one file) breaks.
            if self._segment_size and self._segment_size + len(payload) > (
                self._segment_bytes
            ):
                try:
                    # fsync the segment being retired; it will not be
                    # appended to again.
                    self._write_bytes(self._segment_path(), b"", sync=True)
                except Exception:  # noqa: BLE001 — durability is best-effort
                    logger.opt(exception=True).warning("run log: segment fsync failed")
                self._segment_index += 1
                self._segment_size = 0
            try:
                self._write_bytes(self._segment_path(), payload)
            except Exception:
                logger.opt(exception=True).warning(
                    "run log: append failed; logging disabled for this run"
                )
                self._active = False
                return None
            self._segment_size += len(payload)
            return RunLogRecordNumber(record.number, truncated=bool(truncated_from))

    def write_manifest(self, metadata: dict) -> None:
        """Write run-level convenience metadata. Never raises.

        The manifest is deliberately NOT load-bearing: segment discovery is
        glob + sort (``run_log_search.load_records``), so a crashed run that
        never reaches this call is still fully readable.

        Args:
            metadata: Run-level fields (model, budget, status, supersession).
        """
        if self.log_dir is None:
            return
        import json

        payload = dict(metadata)
        try:
            payload["segments"] = [p.name for p in sorted(self.log_dir.glob("logs.*.txt"))]
            payload["record_count"] = self._counter
            self._write_bytes(
                self.log_dir / MANIFEST_NAME,
                json.dumps(payload, indent=2, default=str).encode("utf-8"),
                sync=True,
            )
        except Exception:  # noqa: BLE001 — convenience metadata only
            logger.opt(exception=True).warning("run log: manifest write failed")

    def close(self) -> None:
        """Flush the final segment to disk. Idempotent and always safe."""
        if not self._active or self.log_dir is None:
            return
        try:
            self._write_bytes(self._segment_path(), b"", sync=True)
        except Exception:  # noqa: BLE001 — best-effort durability
            logger.opt(exception=True).warning("run log: final fsync failed")


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
