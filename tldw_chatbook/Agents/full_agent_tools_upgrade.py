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
  ``spawn_subagent`` creates). A call the loop refused -- its own
  ``tool_result`` is a refusal, such as the light surface's "Tool not
  permitted" for a tool a small model named anyway -- was never dispatched
  and is not use (review round 1, F3-11; paired by position, not ``call_id``
  alone, review v5r1 R1-2). Nor is a call to the runtime's own scaffolding,
  tool discovery and the run's own log and inbox
  (``RUNTIME_SCAFFOLDING_TOOL_NAMES``, review v5r2 G532-R2-3): it says
  nothing about what the user uses, and a tool it loads that then runs
  records its own call.

Runs are read from the Console's ``agent_runs.db`` beside ChaChaNotes and,
when it exists, the served mobile companion's ``<user_data>/mobile/runs.db``
(review round 1, F5-6): phone sends got the full agent surface before the
light surface too. When ChaChaNotes is unavailable the Console's runs
database cannot be found, and that is a failure, not "no signal" (F3-4).

How it runs: in the app (``run_for_profile``), once per launch, as the second
staggered boot worker, right after Actor Pack recovery (which stays first), on
a worker thread after the UI is ready, so it never blocks startup or the UI
loop; and in ``tldw-serve`` (``run_for_served_profile``), at every start,
before the server accepts a connection, so it finishes or fails closed before
the mobile companion can plan a phone send (review v5r2, G532-R2-1). In the
app a Console send planned before the worker lands gets the light set, the
safe default (``Utils/boot_worker_policy.py``). The window is recorded, not
closed. The check's reads start when recovery frees the boot tier's one slot.
Measured live at load ~20, that was 0.7-1.3 s after the staggered gate opened
on a relaunched profile, and 2.8 s on a fresh profile's first launch. A send
in that window could feed the check only by finishing a model round trip and
committing a dispatched call to a tool outside the light set first. On the
light surface that can only be a tool the user turned on themselves: a
built-in with its own gate on, Library access, an MCP server they added or an
armed shell (``Agents/plain_chat_tool_surface.py``).
Both run the same steps, in order:

1. A recorded marker means the upgrade already ran for this profile: done.
   The marker is a row in the workspaces database's one-time backfill table,
   beside the workspace agent backfill's -- never config.toml, because a
   launch that changes no setting must leave config.toml byte-for-byte as it
   found it (TASK-35835/TASK-36385).
2. A config that already sets ``full_agent_tools_enabled`` (true or false) is
   left exactly as it is; the marker is recorded.
3. The signals are read cheaply and read-only (the runs database through the
   private SQLite seam, never created or migrated), under a time budget.
4. Signals found: one line, ``full_agent_tools_enabled = true``, is added
   under ``[tools]``, re-checked against the raw file under the config write
   lock; every other byte of config.toml (comments, table order, spacing) is
   kept. Then the marker. No signal: only the marker.

Any failure before the line is saved writes nothing and records no marker, so
the profile stays on the light set for this launch and the next launch tries
again. Once the line is saved, a failure after it -- the runtime config
refresh (``ConfigPostCommitError``) or the marker write -- is not "nothing
written" (pre-merge review 2026-11-12): the run reports ``KEPT_ON`` with the
failed step, the app mirrors it, and the next launch reads the saved value
as explicit and records the marker (step 2). Logs name the outcome, the
first signal found, a failed step and an exception type; never a path, a
tool name or other user content.

A lost marker (the workspaces database replaced, rebuilt or restored without
it) makes the next launch run the upgrade again. A rerun writes only when the
raw config has no ``full_agent_tools_enabled`` at all and only ``true``, so a
user who turned Full agent tools off (the key is explicitly ``false``) is
never switched back on (step 2). It is NOT limited to agent use from before
the light surface, though (review v5r1 R1-3, correcting round 1's F4-5
"safe by construction"): a light profile that never opted in can since have
dispatched non-light calls without Full agent tools -- a tool switched on
one by one, a tool from an MCP server the user added, a Library tool, a
``$skill`` turn -- or saved a sub-agent definition, and a rerun reads those
as agent use and turns Full agent tools on. Nothing persisted outside the
workspaces database can tell a rerun from a first run without writing
config.toml for a profile with no signal (AC#2 forbids that). Ruled in
review v5r2 (G532-R2-2), with no second marker: the check runs once per
profile as recorded in its workspaces database, and a database replaced
without that record is checked again (AC#3 says so). An explicit value is
never changed, a restored pre-upgrade database is a pre-upgrade profile,
and the scaffolding exclusion above narrows what a rerun can count. A
first run that failed retries on the next launch's signals the same way.
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

from tldw_chatbook.Agents.agent_models import (
    FIND_TOOLS_NAME,
    LOAD_TOOLS_NAME,
    MESSAGE_TOOL_NAMES,
    READ_AGENT_MESSAGES_TOOL_NAME,
    RUN_LOG_SLICE_TOOL_NAME,
    RUN_LOG_STATS_TOOL_NAME,
    SEARCH_RUN_LOG_TOOL_NAME,
)
from tldw_chatbook.Agents.plain_chat_tool_surface import (
    FULL_AGENT_TOOLS_GATE_KEY,
    LIGHT_SURFACE_TOOL_NAMES,
)

__all__ = [
    "FULL_AGENT_TOOLS_UPGRADE_MARKER",
    "RUNTIME_SCAFFOLDING_TOOL_NAMES",
    "SIGNAL_AGENT_DEFINITION",
    "SIGNAL_SUBAGENT_RUN",
    "SIGNAL_TOOL_CALL",
    "SIGNAL_WORKSPACE_BINDING",
    "STEP_MARKER",
    "STEP_RUNTIME_REFRESH",
    "UpgradeOutcome",
    "UpgradeResult",
    "detect_agent_feature_use",
    "run_for_profile",
    "run_for_served_profile",
    "run_full_agent_tools_upgrade",
]

#: The key this upgrade records in the workspaces database's one-time
#: backfill table (``WorkspaceDB.mark_agent_backfill_complete``).
FULL_AGENT_TOOLS_UPGRADE_MARKER = "full_agent_tools_upgrade"

#: The Console's runs database file, beside ChaChaNotes (``ConsoleRuntime``).
_AGENT_RUNS_DB_NAME = "agent_runs.db"

#: The client id ``tldw-serve``'s run opens the profile's workspaces database
#: with (``run_for_served_profile``).
_SERVED_CLIENT_ID = "full-agent-tools-upgrade"

#: How long the signal reads may take before they give up (a failure: the
#: profile stays light and the next launch retries). Every read is over a
#: small table except the step-log scan, which a heavy profile can make long
#: (measured: about 1 s for 200,000 steps on a loaded machine); the budget
#: bounds it. The scan stops at the first non-light call, so only a profile
#: with a very long history and no such call nears it. Such a profile pays
#: at most this on a worker thread at each launch, holding the staggered
#: boot tier's one slot (``Utils/boot_worker_policy.py``) -- never the UI.
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

#: The agent runtime's own scaffolding, which is not "using agent features"
#: (review v5r2, G532-R2-3/LV2-1): tool discovery (``find_tools``,
#: ``load_tools``) and the run reading its own log (``search_run_log``,
#: ``run_log_stats``, ``run_log_slice``) or its own inbox
#: (``read_agent_messages``). The full surface offered these on every send
#: before the light surface, the loop dispatches the discovery pair even on
#: the light surface, and small models call them reflexively: live, a 0.5B
#: model called find_tools and load_tools for "Reply with just: hi". No real
#: use is lost: a tool that discovery loads and the model then calls records
#: its own ``tool_call`` step, which counts. The runtime names no such set
#: (``RUNTIME_TOOL_NAMES`` also holds spawning, skills, fleet and chat-fork
#: tools, which are agent features), so it is defined here, once, from the
#: runtime's own name constants.
RUNTIME_SCAFFOLDING_TOOL_NAMES: frozenset[str] = frozenset(
    {
        FIND_TOOLS_NAME,
        LOAD_TOOLS_NAME,
        SEARCH_RUN_LOG_TOOL_NAME,
        RUN_LOG_STATS_TOOL_NAME,
        RUN_LOG_SLICE_TOOL_NAME,
        READ_AGENT_MESSAGES_TOOL_NAME,
    }
)


#: The refusal text the loop records for a call it never started
#: (``agent_runtime``: ``Tool not permitted: <name>``), with the loop's
#: ``ERROR: `` prefix.
_REFUSED_RESULT_TEXT = "ERROR: Tool not permitted:"


class UpgradeOutcome(str, enum.Enum):
    """What one run of the upgrade did."""

    #: The marker was already recorded: nothing read, nothing written.
    ALREADY_RECORDED = "already_recorded"
    #: The config sets the key itself: left exactly as it is.
    EXPLICIT = "explicit"
    #: Agent features in use: ``full_agent_tools_enabled = true`` written.
    #: Also when a step after the write failed (``unfinished_step``): the
    #: line is saved, so Full agent tools is on, and the next launch reads
    #: it as an explicit value and records the marker.
    KEPT_ON = "kept_on"
    #: No agent feature in use: the light set stays, config untouched.
    LIGHT = "light"
    #: Detection or a write failed: nothing written, retried next launch.
    FAILED = "failed"


#: ``UpgradeResult.unfinished_step`` values: the line was saved, then the
#: runtime config refresh (``ConfigPostCommitError``) or the marker write
#: failed (pre-merge review 2026-11-12).
STEP_RUNTIME_REFRESH = "runtime_refresh"
STEP_MARKER = "marker"


@dataclass(frozen=True)
class UpgradeResult:
    """One run's outcome, the first signal found, and why a run failed.

    ``error_type`` names the exception of a ``FAILED`` run, or of the step
    after the write that failed in a ``KEPT_ON`` run (``unfinished_step``).
    """

    outcome: UpgradeOutcome
    signal: str | None = None
    error_type: str | None = None
    unfinished_step: str | None = None


class _DetectionTimeout(Exception):
    """The signal reads ran past their budget."""


class _RunsDatabaseUnavailable(Exception):
    """The Console's runs database cannot be located (no usable ChaChaNotes).

    Not "no signal": a profile whose only signal lives there would lose its
    tools for good if this launch recorded the marker (review round 1, F3-4).
    """


def _deadline_handler(deadline: float) -> Callable[[], int]:
    """An SQLite progress handler that interrupts the query past ``deadline``."""

    def handler() -> int:
        return int(time.monotonic() >= deadline)

    return handler


def _names(rows: Any) -> set[str]:
    return {str(row[1]) for row in rows}


def _named_list(prefix: str, values: Collection[str], parameters: dict) -> str:
    """Bind ``values`` as named SQL parameters; return their placeholders."""
    names = []
    for index, value in enumerate(values):
        parameters[f"{prefix}{index}"] = value
        names.append(f":{prefix}{index}")
    return ", ".join(names)


def _valid_json(column: str) -> str:
    return f"(CASE WHEN json_valid({column}) THEN {column} ELSE '{{}}' END)"


def _own_result_refused(
    *, results: str, result: str, between: str, other: str, messages: str
) -> str:
    """SQL: the call ``c`` was refused by the loop, so it is not use.

    Its OWN result -- the first ``tool_result`` carrying its ``call_id`` and
    tool after it, with no other call or result carrying that id in between
    -- is a refusal. Pairing by position matters because ``call_id`` is not unique
    in a run: a server that omits ids gets ``call_<position>`` afresh each
    turn (``native_tools.ensure_tool_call_ids``), review v5r1 R1-2. A refusal
    is the loop's ``blocked`` outcome, the not-started "Tool not permitted"
    some runtime tools record as ``failed`` (``send_to_agent`` with no
    fleet), or any failure of a fleet message tool (each is a bounded
    refusal code; nothing ran). A call with no ``call_id`` cannot be paired
    and counts.
    """

    def field(step: str, name: str) -> str:
        return f"json_extract({step}, '$.{name}')"

    call_id = field("c.step", "call_id")
    return (
        f"COALESCE({call_id}, '') <> '' AND EXISTS (SELECT 1 FROM {results} "
        f"AND {field(result, 'kind')} = 'tool_result' "
        f"AND {field(result, 'call_id')} = {call_id} "
        f"AND {field(result, 'tool_name')} = {field('c.step', 'tool_name')} "
        f"AND ({field(result, 'tool_outcome')} = 'blocked' "
        f"OR instr(COALESCE({field(result, 'result')}, ''), :refused_text) = 1 "
        f"OR ({field(result, 'tool_outcome')} = 'failed' "
        f"AND {field(result, 'tool_name')} IN ({messages}))) "
        f"AND NOT EXISTS (SELECT 1 FROM {between} "
        f"AND {field(other, 'call_id')} = {call_id} "
        f"AND {field(other, 'kind')} IN ('tool_call', 'spawn', 'tool_result')))"
    )


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
    parameters: dict[str, str] = {"refused_text": _REFUSED_RESULT_TEXT}
    kinds = _named_list("run_kind", _AGENT_RUN_KINDS, parameters)
    step_kinds = _named_list("step_kind", _TOOL_STEP_KINDS, parameters)
    not_use = _named_list(
        "not_use",
        ("", *light_names, *sorted(RUNTIME_SCAFFOLDING_TOOL_NAMES)),
        parameters,
    )
    messages = _named_list("message", sorted(MESSAGE_TOOL_NAMES), parameters)
    # The JSON is checked before it is parsed: one malformed row must not
    # turn the whole read into an error. The instr() pre-filters skip the
    # parse for the (many) steps that cannot be a tool call or a result.
    tool_step = (
        f"json_extract(c.step, '$.kind') IN ({step_kinds}) "
        f"AND COALESCE(json_extract(c.step, '$.tool_name'), '') NOT IN ({not_use})"
    )
    if "agent_run_steps" in tables:
        query = (
            "SELECT 1 FROM ("
            "SELECT s.run_id AS run_id, s.seq AS pos, "
            f"{_valid_json('s.payload')} AS step FROM agent_run_steps AS s "
            "JOIN agent_runs AS r ON r.id = s.run_id "
            f"WHERE r.agent_kind IN ({kinds}) "
            "AND (instr(s.payload, 'tool_call') > 0 OR instr(s.payload, 'spawn') > 0)"
            f") AS c WHERE {tool_step} AND NOT ("
            + _own_result_refused(
                results="agent_run_steps AS b WHERE b.run_id = c.run_id "
                "AND b.seq > c.pos AND instr(b.payload, 'tool_result') > 0",
                result=_valid_json("b.payload"),
                between="agent_run_steps AS m WHERE m.run_id = c.run_id "
                "AND m.seq > c.pos AND m.seq < b.seq",
                other=_valid_json("m.payload"),
                messages=messages,
            )
            + ") LIMIT 1"
        )
        if connection.execute(query, parameters).fetchone() is not None:
            return SIGNAL_TOOL_CALL
    if "steps" in run_columns:
        query = (
            "SELECT 1 FROM ("
            "SELECT run.blob AS blob, step.key AS pos, step.value AS step FROM ("
            "SELECT CASE WHEN json_valid(r.steps) AND json_type(r.steps) = 'array' "
            "THEN r.steps ELSE '[]' END AS blob FROM agent_runs AS r "
            f"WHERE r.agent_kind IN ({kinds}) "
            "AND (instr(r.steps, 'tool_call') > 0 OR instr(r.steps, 'spawn') > 0)"
            ") AS run, json_each(run.blob) AS step WHERE step.type = 'object'"
            f") AS c WHERE {tool_step} AND NOT ("
            + _own_result_refused(
                results="json_each(c.blob) AS b WHERE b.type = 'object' "
                "AND b.key > c.pos",
                result="b.value",
                between="json_each(c.blob) AS m WHERE m.type = 'object' "
                "AND m.key > c.pos AND m.key < b.key",
                other="m.value",
                messages=messages,
            )
            + ") LIMIT 1"
        )
        if connection.execute(query, parameters).fetchone() is not None:
            return SIGNAL_TOOL_CALL
    return None


def _read_runs_signal(
    path: Path, deadline: float, light_names: tuple[str, ...]
) -> str | None:
    """Open one existing runs database read-only and read its signal."""
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    with closing(
        connect_private_sqlite(
            "agents.full_agent_tools_upgrade",
            path,
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
            return _runs_signal(connection, light_names)
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise _DetectionTimeout() from exc
            raise


def detect_agent_feature_use(
    agent_runs_path: Path | None,
    workspace_db: Any,
    *,
    companion_runs_path: Path | Callable[[], Path | None] | None = None,
    budget_seconds: float = DETECTION_BUDGET_SECONDS,
    light_names: Collection[str] = LIGHT_SURFACE_TOOL_NAMES,
) -> str | None:
    """Return the first agent-feature signal this profile shows, or ``None``.

    Args:
        agent_runs_path: The Console's ``agent_runs.db``. An absent file has
            no signal; the file is never created, migrated or written.
            ``None`` means its location is unknown, which is a failure.
        workspace_db: The profile's ``WorkspaceDB`` (bindings live there).
        companion_runs_path: The served mobile companion's runs database
            (or a callable resolving it, called only when needed). Read the
            same way when it exists.
        budget_seconds: How long the reads may take in all.
        light_names: The light set; a tool call to one of these is not a
            signal (nor is one to ``RUNTIME_SCAFFOLDING_TOOL_NAMES``).

    Returns:
        ``SIGNAL_WORKSPACE_BINDING``, ``SIGNAL_AGENT_DEFINITION``,
        ``SIGNAL_SUBAGENT_RUN`` or ``SIGNAL_TOOL_CALL``, or ``None``.

    Raises:
        Exception: Any read failure, including ``_DetectionTimeout`` past the
            budget and ``_RunsDatabaseUnavailable`` for an unknown runs
            database. The caller treats every failure as "stay light".
    """
    deadline = time.monotonic() + budget_seconds
    if workspace_db.has_runtime_binding(_WORKSPACE_TOOL_BINDING_KINDS):
        return SIGNAL_WORKSPACE_BINDING
    if agent_runs_path is None:
        raise _RunsDatabaseUnavailable()
    if callable(companion_runs_path):
        companion_runs_path = companion_runs_path()
    for path in (agent_runs_path, companion_runs_path):
        if path is None or not Path(path).is_file():
            continue
        if time.monotonic() >= deadline:
            raise _DetectionTimeout()
        signal = _read_runs_signal(Path(path), deadline, tuple(light_names))
        if signal is not None:
            return signal
    return None


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
    """Add the one line ``full_agent_tools_enabled = true`` under ``[tools]``.

    Every other byte of config.toml is kept: the user's comments, table
    order and spacing (review round 1, F1-10 -- the settings writer
    re-serialises the whole file). The raw file is re-read under the config
    write lock, so a value the user set meanwhile (true or false) is never
    overwritten.

    Returns:
        ``True`` when the key was written; ``False`` when the raw file already
        set it by the time the lock was held (nothing written).

    Raises:
        ConfigPostCommitError: The line was saved, but the runtime config
            refresh after it failed.
        Exception: The write did not land, or the file cannot take the line
            without a rewrite (``ConfigTextInsertError``); nothing written.
    """
    from tldw_chatbook.config import insert_absent_cli_config_bool

    return insert_absent_cli_config_bool("tools", FULL_AGENT_TOOLS_GATE_KEY, True)


def run_full_agent_tools_upgrade(
    *,
    workspace_db: Any,
    agent_runs_path: Path | None,
    companion_runs_path: Path | Callable[[], Path | None] | None = None,
    budget_seconds: float = DETECTION_BUDGET_SECONDS,
) -> UpgradeResult:
    """Run the one-time upgrade for one profile (see the module docstring).

    Args:
        workspace_db: The profile's ``WorkspaceDB``: the bindings it holds and
            where the marker is recorded. ``None`` fails (nothing written).
        agent_runs_path: The profile's ``agent_runs.db``; ``None`` (location
            unknown) fails unless the marker or an explicit value decides.
        companion_runs_path: The served companion's runs database, or a
            callable resolving it when detection needs it.
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
            agent_runs_path,
            workspace_db,
            companion_runs_path=companion_runs_path,
            budget_seconds=budget_seconds,
        )
        if signal is None:
            outcome = UpgradeOutcome.LIGHT
        else:
            from tldw_chatbook.config import ConfigPostCommitError

            try:
                written = _write_full_agent_tools_on()
            except ConfigPostCommitError as exc:  # the line is saved
                return _logged(_kept_on_unfinished(signal, STEP_RUNTIME_REFRESH, exc))
            outcome = UpgradeOutcome.KEPT_ON if written else UpgradeOutcome.EXPLICIT
        try:
            workspace_db.mark_agent_backfill_complete(FULL_AGENT_TOOLS_UPGRADE_MARKER)
        except Exception as exc:
            if outcome is not UpgradeOutcome.KEPT_ON:
                raise
            return _logged(_kept_on_unfinished(signal, STEP_MARKER, exc))
        return _logged(UpgradeResult(outcome, signal))
    except Exception as exc:  # noqa: BLE001 -- any failure keeps the light set
        return _logged(
            UpgradeResult(UpgradeOutcome.FAILED, error_type=type(exc).__name__)
        )


def _kept_on_unfinished(signal: str, step: str, exc: Exception) -> UpgradeResult:
    """KEPT_ON whose ``step`` after the write failed (pre-merge review
    2026-11-12): the line is on disk, so it is reported, mirrored into the
    app and logged as on, never as "nothing written"."""
    return UpgradeResult(
        UpgradeOutcome.KEPT_ON, signal, type(exc).__name__, step
    )


def _logged(result: UpgradeResult) -> UpgradeResult:
    if result.outcome is UpgradeOutcome.FAILED:
        logger.warning(
            "Full agent tools upgrade check failed; plain chat keeps the light "
            "tool set and the next launch retries (error_type={})",
            result.error_type,
        )
    elif result.unfinished_step is not None:
        logger.warning(
            "Full agent tools upgrade check: outcome={} signal={}; "
            "full_agent_tools_enabled = true is saved in config.toml, but the "
            "step after it failed (step={}, error_type={}); the next launch "
            "finishes the check",
            result.outcome.value,
            result.signal or "none",
            result.unfinished_step,
            result.error_type,
        )
    else:
        logger.info(
            "Full agent tools upgrade check: outcome={} signal={}",
            result.outcome.value,
            result.signal or "none",
        )
    return result


def _agent_runs_path(chachanotes_db: Any) -> Path | None:
    """The Console's runs database: ``agent_runs.db`` beside ChaChaNotes.

    The derivation every production reader uses (``ConsoleRuntime``,
    Settings > Agents, change retention). ``None`` when ChaChaNotes is
    unavailable or in memory: the location is unknown, a failure.
    """
    db_path = getattr(chachanotes_db, "db_path", None)
    if not db_path or str(db_path) == ":memory:" or str(db_path).startswith("file:"):
        return None
    return Path(db_path).parent / _AGENT_RUNS_DB_NAME


def _companion_runs_path() -> Path:
    """The served mobile companion's runs database.

    ``Web_Server/serve.py`` builds the companion over
    ``get_user_data_dir() / "mobile"`` and ``MobileChatService`` keeps its
    runs in ``runs.db`` there.
    """
    from tldw_chatbook.config import get_user_data_dir

    return Path(get_user_data_dir()) / "mobile" / "runs.db"


def _mirror_into_app_config(app_config: Any) -> None:
    """Keep the app's in-memory config in step with the written key.

    Setup's Tools step and Summary read ``app.app_config``; the MCP hub's
    gate save mirrors the same way (TASK-34100.5.3 review, B-F5).
    """
    if not isinstance(app_config, dict):
        return
    tools = app_config.get("tools")
    if not isinstance(tools, dict):
        tools = app_config["tools"] = {}
    tools[FULL_AGENT_TOOLS_GATE_KEY] = True


def run_for_profile(
    *,
    workspace_db: Any,
    chachanotes_db: Any,
    app_config: Any,
    call_from_thread: Callable[..., Any],
) -> UpgradeResult:
    """The boot worker's body: run the upgrade for the app's own profile.

    Runs on a worker thread. Closes the thread's workspaces-database
    connection when done, and mirrors a written key into ``app_config`` on
    the UI thread through ``call_from_thread``.

    Args:
        workspace_db: The app's ``WorkspaceDB`` (``None`` fails closed).
        chachanotes_db: The app's ChaChaNotes database; the runs database
            sits beside it.
        app_config: The app's in-memory config dict.
        call_from_thread: The app's ``call_from_thread``.

    Returns:
        The outcome. Never raises.
    """
    try:
        result = run_full_agent_tools_upgrade(
            workspace_db=workspace_db,
            agent_runs_path=_agent_runs_path(chachanotes_db),
            companion_runs_path=_companion_runs_path,
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
            call_from_thread(_mirror_into_app_config, app_config)
        except Exception:  # noqa: BLE001 -- the app may be closing; config is saved
            pass
    return result


def run_for_served_profile() -> UpgradeResult:
    """``tldw-serve``'s run: the same check, before the server takes a request.

    Review v5r2 (G532-R2-1). The served mobile companion plans phone sends in
    the serve process, and its runs (``<user_data>/mobile/runs.db``) are one
    of the check's sources. Run only by the app, the check could not tell a
    serve-first profile's light-era phone use from use before the light
    surface: a fresh profile first used from the phone, with one tool
    switched on by itself, had Full agent tools turned on at its first app
    launch. ``Web_Server.serve.run_web_server`` calls this before
    ``serve()``, so it finishes, or fails closed, before the listener binds
    and so before any phone send can be planned. A fresh profile is recorded
    while it has no signal; a phone-only profile from before the light
    surface is upgraded at serve start, and the companion's next send reads
    the new value (the write republishes the in-process config).

    It opens the profile's own workspaces database (the bindings and the
    marker, the one the app opens; created if absent, as the app would),
    finds the Console's runs database beside ChaChaNotes and the companion's
    in ``<user_data>/mobile``, and closes what it opened.

    Returns:
        The outcome. Never raises.
    """
    try:
        from tldw_chatbook.app_services import open_profile_storage
        from tldw_chatbook.config import (
            get_chachanotes_db_path,
            get_workspaces_db_path,
        )

        agent_runs_path = Path(get_chachanotes_db_path()).parent / _AGENT_RUNS_DB_NAME
        storage = open_profile_storage(
            client_id=_SERVED_CLIENT_ID, workspaces=get_workspaces_db_path()
        )
    except Exception as exc:  # noqa: BLE001 -- any failure keeps the light set
        return _logged(
            UpgradeResult(UpgradeOutcome.FAILED, error_type=type(exc).__name__)
        )
    try:
        return run_full_agent_tools_upgrade(
            workspace_db=storage.workspaces,
            agent_runs_path=agent_runs_path,
            companion_runs_path=_companion_runs_path,
        )
    finally:
        storage.close()  # this thread's connection; never raises (ADR-250 s5)
