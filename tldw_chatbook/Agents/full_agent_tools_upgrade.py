"""Keep Full agent tools on for profiles that already use agent features.

TASK-34100.5.3.2. TASK-34100.5.3 (PR #3295) gave plain Console chat a light
tool surface (date/time, calculator, ask-user) until ``[tools]
full_agent_tools_enabled`` is on, and that key defaults to off for every
profile that lacks it. A profile that existed before the change would lose
what it uses in plain chat today: sub-agents, workspace file and Git tools
(SSH remote workspaces too), web and Watchlists tools, character tools and
Canvas.

Owner ruling, 2026-10-29 ("Keep them on if used"): a one-time migration turns
Full agent tools on for a profile that already uses agent features. Fresh
installs, and profiles that never used them, keep the light set and get
nothing written to their config.

"Uses agent features" is at least one of:

* a user-defined sub-agent definition (a live ``agent_definitions`` row);
* a workspace binding, local folder or SSH remote
  (``workspace_runtime_bindings``);
* an earlier Console agent run that called a tool outside the light set (a
  ``tool_call``/``spawn`` step of a primary or sub-agent run, in the per-row
  step log or the pre-v13 JSON blob, or any sub-agent run, which only
  ``spawn_subagent`` creates).

How it runs (``run_for_app``): once per launch, as the first staggered boot
worker, on a worker thread after the UI is ready, so it never blocks startup
or the UI loop. In order:

1. A recorded marker means the upgrade already ran for this profile: done.
   The marker is a row in the workspaces database's one-time backfill table,
   beside the workspace agent backfill's -- never config.toml, because a
   launch that changes no setting must leave config.toml byte-for-byte as it
   found it (TASK-35835/TASK-36385).
2. A config that already sets ``full_agent_tools_enabled`` (true or false) is
   left exactly as it is; the marker is recorded.
3. The signals are read cheaply and read-only (the runs database through the
   private SQLite seam, never created or migrated), under a time budget.
4. Signals found: one literal config mutation sets the one key, re-checked
   against the raw file under the config write lock. Then the marker.
   No signal: only the marker.

Any failure writes nothing and records no marker, so the profile stays on the
light set for this launch and the next launch tries again. Logs name the
outcome, the first signal found and an exception type; never a path, a tool
name or other user content.
"""

from __future__ import annotations

import enum
import time
import tomllib
from collections.abc import Callable, Collection
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_chatbook.Agents.plain_chat_tool_surface import (
    FULL_AGENT_TOOLS_GATE_KEY,
    LIGHT_SURFACE_TOOL_NAMES,
)

__all__ = [
    "FULL_AGENT_TOOLS_UPGRADE_MARKER",
    "SIGNAL_AGENT_DEFINITION",
    "SIGNAL_SUBAGENT_RUN",
    "SIGNAL_TOOL_CALL",
    "SIGNAL_WORKSPACE_BINDING",
    "UpgradeOutcome",
    "UpgradeResult",
    "detect_agent_feature_use",
    "run_for_app",
    "run_full_agent_tools_upgrade",
]

#: The key this upgrade records in the workspaces database's one-time
#: backfill table (``WorkspaceDB.mark_agent_backfill_complete``).
FULL_AGENT_TOOLS_UPGRADE_MARKER = "full_agent_tools_upgrade"

#: How long the signal reads may take before they give up (a failure: the
#: profile stays light and the next launch retries). Every read is over a
#: small table except the step-log scan, which a heavy profile can make long
#: (measured: about 1 s for 200,000 steps on a loaded machine); the budget
#: bounds it.
DETECTION_BUDGET_SECONDS = 5.0

#: SQLite VM steps between budget checks: often enough to stop a long scan
#: promptly, rare enough to cost nothing on the short reads.
_BUDGET_CHECK_STEPS = 1000

SIGNAL_WORKSPACE_BINDING = "workspace_binding"
SIGNAL_AGENT_DEFINITION = "agent_definition"
SIGNAL_SUBAGENT_RUN = "subagent_run"
SIGNAL_TOOL_CALL = "tool_call"

#: Binding kinds that give the Console workspace tools: a local folder and an
#: SSH remote directory (``Workspaces.models.RuntimeBindingKind``).
_WORKSPACE_TOOL_BINDING_KINDS: tuple[str, ...] = ("local-filesystem", "ssh-filesystem")

#: Agent kinds whose runs are the Console agent's own (a local ``!command``
#: run is the user's shell, not a model tool call).
_AGENT_RUN_KINDS: tuple[str, ...] = ("primary", "subagent")

#: Step kinds that record a model's tool call (``Agents.agent_models``).
_TOOL_STEP_KINDS: tuple[str, ...] = ("tool_call", "spawn")


class UpgradeOutcome(str, enum.Enum):
    """What one run of the upgrade did."""

    #: The marker was already recorded: nothing read, nothing written.
    ALREADY_RECORDED = "already_recorded"
    #: The config sets the key itself: left exactly as it is.
    EXPLICIT = "explicit"
    #: Agent features in use: ``full_agent_tools_enabled = true`` written.
    KEPT_ON = "kept_on"
    #: No agent feature in use: the light set stays, config untouched.
    LIGHT = "light"
    #: Detection or a write failed: nothing written, retried next launch.
    FAILED = "failed"


@dataclass(frozen=True)
class UpgradeResult:
    """One run's outcome, the first signal found, and why a run failed."""

    outcome: UpgradeOutcome
    signal: str | None = None
    error_type: str | None = None


class _DetectionTimeout(Exception):
    """The signal reads ran past their budget."""


def _deadline_handler(deadline: float) -> Callable[[], int]:
    """An SQLite progress handler that interrupts the query past ``deadline``."""

    def handler() -> int:
        return int(time.monotonic() >= deadline)

    return handler


def _names(rows: Any) -> set[str]:
    return {str(row[1]) for row in rows}


def _runs_signal(connection: Any, light_names: tuple[str, ...]) -> str | None:
    """The first agent-feature signal in an open runs database, or ``None``."""
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if "agent_definitions" in tables:
        definition_columns = _names(
            connection.execute("PRAGMA table_info(agent_definitions)")
        )
        query = (
            "SELECT 1 FROM agent_definitions WHERE deleted = 0 LIMIT 1"
            if "deleted" in definition_columns
            else "SELECT 1 FROM agent_definitions LIMIT 1"
        )
        if connection.execute(query).fetchone() is not None:
            return SIGNAL_AGENT_DEFINITION
    if "agent_runs" not in tables:
        return None
    run_columns = _names(connection.execute("PRAGMA table_info(agent_runs)"))
    if "agent_kind" not in run_columns:
        return None
    if (
        connection.execute(
            "SELECT 1 FROM agent_runs WHERE agent_kind = 'subagent' LIMIT 1"
        ).fetchone()
        is not None
    ):
        return SIGNAL_SUBAGENT_RUN
    kinds = ", ".join("?" for _ in _AGENT_RUN_KINDS)
    step_kinds = ", ".join("?" for _ in _TOOL_STEP_KINDS)
    light = ", ".join("?" for _ in light_names)
    # The JSON is checked before it is parsed: one malformed row must not
    # turn the whole read into an error. The instr() pre-filter skips the
    # parse for the (many) steps that cannot be a tool call.
    tool_step = (
        f"json_extract(step, '$.kind') IN ({step_kinds}) "
        f"AND COALESCE(json_extract(step, '$.tool_name'), '') "
        f"NOT IN ('', {light})"
    )
    parameters = (*_AGENT_RUN_KINDS, *_TOOL_STEP_KINDS, *light_names)
    if "agent_run_steps" in tables:
        query = (
            "SELECT 1 FROM ("
            "SELECT CASE WHEN json_valid(s.payload) THEN s.payload ELSE '{}' END "
            "AS step FROM agent_run_steps AS s "
            "JOIN agent_runs AS r ON r.id = s.run_id "
            f"WHERE r.agent_kind IN ({kinds}) "
            "AND (instr(s.payload, 'tool_call') > 0 OR instr(s.payload, 'spawn') > 0)"
            f") WHERE {tool_step} LIMIT 1"
        )
        if connection.execute(query, parameters).fetchone() is not None:
            return SIGNAL_TOOL_CALL
    if "steps" in run_columns:
        query = (
            "SELECT 1 FROM ("
            "SELECT step.value AS step FROM agent_runs AS r, json_each("
            "CASE WHEN json_valid(r.steps) AND json_type(r.steps) = 'array' "
            "THEN r.steps ELSE '[]' END) AS step "
            f"WHERE r.agent_kind IN ({kinds}) "
            "AND (instr(r.steps, 'tool_call') > 0 OR instr(r.steps, 'spawn') > 0) "
            "AND step.type = 'object'"
            f") WHERE {tool_step} LIMIT 1"
        )
        if connection.execute(query, parameters).fetchone() is not None:
            return SIGNAL_TOOL_CALL
    return None


def detect_agent_feature_use(
    agent_runs_path: Path | None,
    workspace_db: Any,
    *,
    budget_seconds: float = DETECTION_BUDGET_SECONDS,
    light_names: Collection[str] = LIGHT_SURFACE_TOOL_NAMES,
) -> str | None:
    """Return the first agent-feature signal this profile shows, or ``None``.

    Args:
        agent_runs_path: The profile's ``agent_runs.db``. ``None`` or an
            absent file has no signal; the file is never created, migrated
            or written.
        workspace_db: The profile's ``WorkspaceDB`` (bindings live there).
        budget_seconds: How long the reads may take in all.
        light_names: The light set; a tool call to one of these is not a
            signal.

    Returns:
        ``SIGNAL_WORKSPACE_BINDING``, ``SIGNAL_AGENT_DEFINITION``,
        ``SIGNAL_SUBAGENT_RUN`` or ``SIGNAL_TOOL_CALL``, or ``None``.

    Raises:
        Exception: Any read failure, including ``_DetectionTimeout`` past the
            budget. The caller treats every failure as "stay light".
    """
    deadline = time.monotonic() + budget_seconds
    if workspace_db.has_runtime_binding(_WORKSPACE_TOOL_BINDING_KINDS):
        return SIGNAL_WORKSPACE_BINDING
    if agent_runs_path is None or not Path(agent_runs_path).is_file():
        return None
    if time.monotonic() >= deadline:
        raise _DetectionTimeout()
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    with closing(
        connect_private_sqlite(
            "agents.full_agent_tools_upgrade",
            agent_runs_path,
            read_only=True,
            must_exist=True,
            operation_deadline=deadline,
        )
    ) as connection:
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.set_progress_handler(
            _deadline_handler(deadline), _BUDGET_CHECK_STEPS
        )
        try:
            return _runs_signal(connection, tuple(light_names))
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise _DetectionTimeout() from exc
            raise


def _explicit_in(raw: Any) -> bool:
    """Whether a raw config mapping sets ``[tools] full_agent_tools_enabled``."""
    tools = raw.get("tools") if isinstance(raw, dict) else None
    return isinstance(tools, dict) and FULL_AGENT_TOOLS_GATE_KEY in tools


def _config_sets_key_explicitly() -> bool:
    """Read the raw config file (no defaults merged) for the key.

    Raises:
        Exception: The file could not be read or parsed.
    """
    from tldw_chatbook.config import read_cli_config_snapshot

    serialized = read_cli_config_snapshot().serialized
    if serialized is None:
        return False
    return _explicit_in(tomllib.loads(serialized))


def _write_full_agent_tools_on() -> bool:
    """Set ``[tools] full_agent_tools_enabled = true`` and nothing else.

    The builder re-reads the raw file under the config write lock, so a value
    the user set meanwhile (true or false) is never overwritten.

    Returns:
        ``True`` when the key was written; ``False`` when the raw file already
        set it by the time the lock was held (nothing written).

    Raises:
        RuntimeError: The write did not land.
    """
    from tldw_chatbook.config import (
        LiteralSettingsMutation,
        apply_literal_settings_transaction_to_cli_config,
    )

    explicit_under_lock = False

    def build(snapshot: Any) -> Any:
        nonlocal explicit_under_lock
        explicit_under_lock = _explicit_in(dict(snapshot.raw_values))
        if explicit_under_lock:
            return LiteralSettingsMutation(section_values={}, delete_keys={})
        return LiteralSettingsMutation(
            section_values={("tools",): {FULL_AGENT_TOOLS_GATE_KEY: True}},
            delete_keys={},
        )

    result = apply_literal_settings_transaction_to_cli_config(build)
    if explicit_under_lock:
        return False
    if not (result.file_replaced and result.caches_reloaded):
        raise RuntimeError(f"config write did not land ({result.failure_phase})")
    return True


def run_full_agent_tools_upgrade(
    *,
    workspace_db: Any,
    agent_runs_path: Path | None,
    budget_seconds: float = DETECTION_BUDGET_SECONDS,
) -> UpgradeResult:
    """Run the one-time upgrade for one profile (see the module docstring).

    Args:
        workspace_db: The profile's ``WorkspaceDB``: the bindings it holds and
            where the marker is recorded. ``None`` fails (nothing written).
        agent_runs_path: The profile's ``agent_runs.db``, or ``None``.
        budget_seconds: The signal reads' time budget.

    Returns:
        The outcome. Never raises.
    """
    try:
        if workspace_db is None:
            raise RuntimeError("no workspaces database")
        if workspace_db.is_agent_backfill_complete(FULL_AGENT_TOOLS_UPGRADE_MARKER):
            return UpgradeResult(UpgradeOutcome.ALREADY_RECORDED)
        if _config_sets_key_explicitly():
            workspace_db.mark_agent_backfill_complete(FULL_AGENT_TOOLS_UPGRADE_MARKER)
            return _logged(UpgradeResult(UpgradeOutcome.EXPLICIT))
        signal = detect_agent_feature_use(
            agent_runs_path, workspace_db, budget_seconds=budget_seconds
        )
        if signal is None:
            outcome = UpgradeOutcome.LIGHT
        elif _write_full_agent_tools_on():
            outcome = UpgradeOutcome.KEPT_ON
        else:
            outcome = UpgradeOutcome.EXPLICIT
        workspace_db.mark_agent_backfill_complete(FULL_AGENT_TOOLS_UPGRADE_MARKER)
        return _logged(UpgradeResult(outcome, signal))
    except Exception as exc:  # noqa: BLE001 -- any failure keeps the light set
        return _logged(
            UpgradeResult(UpgradeOutcome.FAILED, error_type=type(exc).__name__)
        )


def _logged(result: UpgradeResult) -> UpgradeResult:
    if result.outcome is UpgradeOutcome.FAILED:
        logger.warning(
            "Full agent tools upgrade check failed; plain chat keeps the light "
            "tool set and the next launch retries (error_type={})",
            result.error_type,
        )
    else:
        logger.info(
            "Full agent tools upgrade check: outcome={} signal={}",
            result.outcome.value,
            result.signal or "none",
        )
    return result


def _agent_runs_path(app: Any) -> Path | None:
    """The Console's runs database: ``agent_runs.db`` beside ChaChaNotes."""
    db_path = getattr(getattr(app, "chachanotes_db", None), "db_path", None)
    if not db_path or str(db_path) == ":memory:" or str(db_path).startswith("file:"):
        return None
    return Path(db_path).parent / "agent_runs.db"


def _mirror_into_app_config(app: Any) -> None:
    """Keep the app's in-memory config in step with the written key.

    Setup's Tools step and Summary read ``app.app_config``; the MCP hub's
    gate save mirrors the same way (TASK-34100.5.3 review, B-F5).
    """
    config = getattr(app, "app_config", None)
    if not isinstance(config, dict):
        return
    tools = config.get("tools")
    if not isinstance(tools, dict):
        tools = config["tools"] = {}
    tools[FULL_AGENT_TOOLS_GATE_KEY] = True


def run_for_app(app: Any) -> UpgradeResult:
    """The boot worker's body: run the upgrade for the app's own profile.

    Runs on a worker thread. Closes the thread's workspaces-database
    connection when done, and mirrors a written key into ``app.app_config``
    on the UI thread.
    """
    workspace_db = getattr(app, "local_workspace_db", None)
    try:
        result = run_full_agent_tools_upgrade(
            workspace_db=workspace_db, agent_runs_path=_agent_runs_path(app)
        )
    finally:
        close = getattr(workspace_db, "close", None)
        if callable(close):
            try:
                close()  # this thread's held connection only
            except Exception:  # noqa: BLE001 -- teardown must not raise
                pass
    if result.outcome is UpgradeOutcome.KEPT_ON:
        try:
            app.call_from_thread(_mirror_into_app_config, app)
        except Exception:  # noqa: BLE001 -- the app may be closing; config is saved
            pass
    return result
