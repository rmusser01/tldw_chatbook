# tldw_chatbook/Agents/tool_catalog.py
"""ToolProvider capability interface + registry + builtin provider.

This is the plugin seam: MCP (task-201) and Skills (task-200) register as
providers here — the runtime never changes. May import tool_executor
(wrapping it is this module's job); no UI/DB imports.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    runtime_checkable,
)

from loguru import logger

from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
)
from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS
from tldw_chatbook.Tools.tool_executor import CalculatorTool, DateTimeTool

from .library_rag_tool_provider import LibraryRagToolProvider, RAG_TOOL_NAME
from .library_tool_provider import BuiltinLibraryAuthority, LibraryToolProvider
from .agent_models import (
    AgentDefinition,
    CHECK_AGENTS_TOOL_NAME,
    FIND_TOOLS_RESULT_LIMIT,
    FIND_TOOLS_NAME,
    INSTALL_SKILL_TOOL_NAME,
    LOAD_TOOLS_NAME,
    RUN_LOG_SLICE_TOOL_NAME,
    RUN_LOG_STATS_TOOL_NAME,
    RUN_SKILL_SCRIPT_TOOL_NAME,
    SEARCH_RUN_LOG_TOOL_NAME,
    SEND_TO_AGENT_TOOL_NAME,
    SKILL_FILE_TOOL_NAME,
    SPAWN_TOOL_NAME,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
    WAIT_AGENTS_TOOL_NAME,
)
from .run_context import current_run_id
from .run_log_search import (
    MAX_CROSS_RUN_RUNS,
    MAX_SLICE_RECORDS,
    MAX_STATS_GROUPS,
    STATS_GROUP_BY_FIELDS,
)
# NOTE (boot budget, ADR-097): `run_tool_policy` is annotation-only here
# (`from __future__ import annotations` above); the TYPE_CHECKING import
# keeps the module off the UI-ready census path. The live policy object is
# constructed by its callers (see `Chat/console_agent_bridge.py`).
if TYPE_CHECKING:
    from .run_tool_policy import RunToolPolicy

LIBRARY_RESERVED_TOOL_NAMES: frozenset[str] = frozenset(
    (*LIBRARY_TOOL_DESCRIPTORS.keys(), RAG_TOOL_NAME)
)


class ToolExecutionPolicy(StrEnum):
    """How the agent runtime may stop waiting after a tool starts."""

    BOUNDED_ABANDONABLE = "bounded_abandonable"
    DEFINITIVE_AFTER_START = "definitive_after_start"

SPAWN_TOOL_SCHEMA = ToolSchema(
    id="runtime:spawn_subagent",
    name=SPAWN_TOOL_NAME,
    description=(
        "Delegate a self-contained task to an isolated sub-agent. It sees "
        "only the task text you pass, works on it, and returns a result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Complete, self-contained task description.",
            }
        },
        "required": ["task"],
    },
)


def build_spawn_schema(definitions: Sequence[AgentDefinition]) -> ToolSchema:
    """The spawn tool's schema for THIS run.

    With no definitions, returns ``SPAWN_TOOL_SCHEMA`` itself (identity —
    byte-identical payloads for every pre-definition caller). With
    definitions, adds an OPTIONAL ``agent`` parameter carrying both an
    ``enum`` (native tool-calling) and a prose roster in the description
    (fence-protocol models read prose better than schema; this text rides
    every fence-model turn, which is why AgentDefinition.description is
    hard-capped).
    """
    if not definitions:
        return SPAWN_TOOL_SCHEMA
    roster = "\n".join(
        f"- {d.name} — {d.description}" if d.description else f"- {d.name}"
        for d in definitions
    )
    parameters = {
        "type": "object",
        "properties": {
            # Shallow-copied so no future consumer of the built schema can
            # mutate the module-global SPAWN_TOOL_SCHEMA through this alias.
            "task": dict(SPAWN_TOOL_SCHEMA.parameters["properties"]["task"]),
            "agent": {
                "type": "string",
                "enum": [d.name for d in definitions],
                "description": (
                    "Optional: run the task as one of these named agents "
                    "(omit for a generic sub-agent):\n" + roster
                ),
            },
        },
        "required": ["task"],
    }
    return ToolSchema(
        id=SPAWN_TOOL_SCHEMA.id,
        name=SPAWN_TOOL_SCHEMA.name,
        description=SPAWN_TOOL_SCHEMA.description,
        parameters=parameters,
    )


# Fleet (PR2a Task 6). Pinned together with the spawn schema, and only
# for a run that actually has a fleet coordinator -- a model told it can
# wait on children it can never start has been handed a dead end.
WAIT_AGENTS_SCHEMA = ToolSchema(
    id="runtime:wait_agents",
    name=WAIT_AGENTS_TOOL_NAME,
    description=(
        "Wait for sub-agents you started with spawn_subagent and collect "
        "their results. Omit 'ids' to wait for every sub-agent still "
        "running, or pass the handle ids spawn_subagent returned to wait "
        "for just those. Results that arrive after your final answer are "
        "wasted, so always call this before you answer. When several "
        "results come back together each one is shortened to share this "
        "turn's result budget -- call wait_agents with a single id to get "
        "that one sub-agent's full result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": ("Handle ids to wait for (omit for all of them)."),
            }
        },
        "required": [],
    },
)

CHECK_AGENTS_SCHEMA = ToolSchema(
    id="runtime:check_agents",
    name=CHECK_AGENTS_TOOL_NAME,
    description=(
        "List every sub-agent you have started in this turn with its "
        "handle id, status, and elapsed time. Returns immediately and "
        "never waits -- use wait_agents to actually collect results."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)

# Fleet steering (PR3b Task 2, spec SS6). Pinned and wired under the SAME
# `fleet_active` predicate as the two schemas above: without a live fleet
# there is no mailbox to post into, and a sub-agent never sees it (depth-1:
# children cannot steer each other). The description is the supervisor's
# whole curriculum -- both id vocabularies, the honest delivery latency,
# and spec SS3 invariant 4 (steering never cancels) -- because nothing else
# teaches the model any of it.
SEND_TO_AGENT_SCHEMA = ToolSchema(
    id="runtime:send_to_agent",
    name=SEND_TO_AGENT_TOOL_NAME,
    description=(
        "Send a steering message to a sub-agent that is still running. "
        "'id' accepts either vocabulary: the handle id spawn_subagent "
        "returned (also shown by check_agents), or the run id a "
        "completion notice named. The message is queued and handed to the "
        "sub-agent as a labeled user-role message at its next model turn "
        "-- a sub-agent inside a long tool call sees it late, only after "
        "that call returns. Steering never cancels or restarts the "
        "sub-agent: it keeps its task and its progress, and simply reads "
        "your message as extra direction. Sent to a recently FINISHED "
        "sub-agent, this instead starts a NEW run seeded with its "
        "retained transcript plus your message (the finished run itself "
        "is untouched); the new run costs a spawn slot, and transcripts "
        "are kept in memory only -- not across an app restart."
    ),
    parameters={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": (
                    "Which sub-agent: a handle id (from spawn_subagent or "
                    "check_agents) or a run id (from a completion notice)."
                ),
            },
            "message": {
                "type": "string",
                "description": (
                    "The steering text to deliver. Plain text; must be non-empty."
                ),
            },
        },
        "required": ["id", "message"],
    },
)


FIND_TOOLS_SCHEMA = ToolSchema(
    id="runtime:find_tools",
    name=FIND_TOOLS_NAME,
    description="Search the tool catalog by keyword; returns ids + one-liners.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

LOAD_TOOLS_SCHEMA = ToolSchema(
    id="runtime:load_tools",
    name=LOAD_TOOLS_NAME,
    description=(
        "Call alone in its tool batch. Select full schemas for catalog ids; "
        "accepted ids replace the current catalog tool set, so include every "
        "tool to retain."
    ),
    parameters={
        "type": "object",
        "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["ids"],
    },
)

SKILL_FILE_TOOL_SCHEMA = ToolSchema(
    id="runtime:skill_file",
    name=SKILL_FILE_TOOL_NAME,
    description=(
        "Read a bundled reference file of a skill active in this run. "
        "Args: skill_name (the skill whose bundle to read), path (relative "
        "POSIX path, e.g. references/api.md). Text files only."
    ),
    parameters={
        "type": "object",
        "properties": {
            "skill_name": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["skill_name", "path"],
    },
)

INSTALL_SKILL_TOOL_SCHEMA = ToolSchema(
    id="runtime:install_skill",
    name=INSTALL_SKILL_TOOL_NAME,
    description=(
        "Install a skill from a GitHub repository/tree URL or a direct "
        "https .zip URL. The user is asked to confirm before anything is "
        "downloaded. On success the skill is installed but left pending the "
        "user's review — it cannot run until the user approves it in "
        "Library > Skills. If the repository contains multiple skills, the "
        "tool returns the list of candidates; re-call with a URL that points "
        "at one skill's subdirectory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "A GitHub repo/tree URL or a direct https .zip URL for "
                    "the skill to install."
                ),
            }
        },
        "required": ["url"],
    },
)

RUN_SKILL_SCRIPT_TOOL_SCHEMA = ToolSchema(
    id="runtime:run_skill_script",
    name=RUN_SKILL_SCRIPT_TOOL_NAME,
    description=(
        "Run a script bundled with a trusted skill. The user is asked to "
        "confirm each run unless they have granted this skill standing "
        "permission. The script runs with a scrubbed environment in a "
        "temporary working directory (not the skill's own folder), under CPU "
        "and time limits; only its stdout/stderr and exit code come back, and "
        "any files it writes are discarded. Args: skill_name (the skill that "
        "owns the script), script_path (relative POSIX path, e.g. "
        "scripts/extract.py), args (optional list of string arguments)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "The skill whose bundled script to run.",
            },
            "script_path": {
                "type": "string",
                "description": (
                    "Relative POSIX path of the script inside the skill's "
                    "bundle, e.g. scripts/extract.py."
                ),
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional string arguments passed to the script.",
            },
        },
        "required": ["skill_name", "script_path"],
    },
)


SEARCH_RUN_LOG_TOOL_SCHEMA = ToolSchema(
    id="runtime:search_run_log",
    name=SEARCH_RUN_LOG_TOOL_NAME,
    description=(
        "Search this run's own complete log. Your context holds a truncated "
        "view; the log holds every model turn, tool call, and tool result in "
        "full. Use it to recover a truncated result or recall an earlier step "
        "instead of re-running work. 'contains' (literal substring) and "
        "'pattern' (regular expression, first 500 characters per record "
        "only) both match a record's CONTENT ONLY -- never its metadata. "
        "Use 'tool', 'type', 'status', and 'kind' to filter by metadata "
        "instead -- e.g. to find every call to a specific tool, filter "
        "with 'tool' rather than 'contains', since the tool's name may "
        "never appear inside its own arguments or result. A record's "
        "rendered content is windowed: when 'contains' or 'pattern' is set, "
        "the window is centred on that record's first match; otherwise it "
        "starts at the beginning. When a record is shown only partially, "
        "the render states the character range and total size, and the "
        "'offset' to pass next to keep reading. By default this searches "
        "only THIS run; set 'scope' to also search this conversation's "
        "earlier runs."
    ),
    parameters={
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "description": (
                    "Which run(s) to search. 'run' (default): only this "
                    "run's own log, exactly as before. 'conversation': "
                    "also search this conversation's earlier runs, newest "
                    f"first, up to {MAX_CROSS_RUN_RUNS} runs per call -- "
                    "each hit is labelled with which run it came from and "
                    "whether it is this run. A run whose log cannot be "
                    "found under the current root (e.g. the workspace "
                    "folder was bound, rebound, or unbound since) is "
                    "reported as unavailable rather than silently omitted "
                    "-- the response always states how many runs were "
                    "actually searched vs. could not be located."
                ),
            },
            "contains": {
                "type": "string",
                "description": "Literal substring to find in a record's content "
                "(case-insensitive). Never matches metadata such as the tool "
                "name -- use 'tool' for that.",
            },
            "pattern": {
                "type": "string",
                "description": "Regular expression over a record's content "
                "only; first 500 chars per record.",
            },
            "tool": {"type": "string", "description": "Filter by tool name."},
            "type": {
                "type": "string",
                "description": "Filter by record type: model, tool_call, tool_result.",
            },
            "status": {"type": "string", "description": "Filter: ok or error."},
            "kind": {
                "type": "string",
                "description": "Filter by agent kind: primary or subagent.",
            },
            "from_record": {"type": "integer", "description": "Lowest record number."},
            "to_record": {"type": "integer", "description": "Highest record number."},
            "context": {
                "type": "integer",
                "description": "Records to include either side of each hit.",
            },
            "offset": {
                "type": "integer",
                "description": "Character offset into each record's rendered "
                "content to start from. Use this to page through a record "
                "larger than the render window -- the previous result names "
                "the offset to pass next. Defaults to 0; ignored in favour "
                "of a match-centred window when 'contains' or 'pattern' "
                "matches and no offset is given.",
            },
        },
        "required": [],
    },
)


# Phase 2 (run-log spec §10, task-1271): two more runtime tools that COMPUTE
# over the log rather than only retrieving from it -- registered exactly
# like SEARCH_RUN_LOG_TOOL_SCHEMA above (same runtime-tool pattern, same
# primary-agent-only gate in agent_service.py's `log_active` block).

RUN_LOG_STATS_TOOL_SCHEMA = ToolSchema(
    id="runtime:run_log_stats",
    name=RUN_LOG_STATS_TOOL_NAME,
    description=(
        "Get bounded aggregate statistics over this run's own log WITHOUT "
        "paging individual records through your context -- counts, error "
        "counts, and content-byte totals grouped by tool, record type, "
        "status, or agent kind (primary vs. sub-agent). Use this to answer "
        "'which tool have I called most, and how often did it fail?' "
        "Output is one line per distinct group value, never one line per "
        "record, so it stays small no matter how long this run gets. At "
        f"most {MAX_STATS_GROUPS} groups are shown per call, ranked by "
        "count (the most frequent survive); if more distinct values "
        "exist, the response says so explicitly with a count of how many "
        "were omitted -- narrow with tool=/type=/status=/kind= or a "
        "record range to see the rest. "
        "Per-record token counts are not tracked in this run's log -- only "
        "the whole run's total token spend is recorded once the run "
        "finishes -- so this tool cannot report a live token total; "
        "content_bytes is reported instead as an exact, always-available "
        "proxy for how much content each group has produced."
    ),
    parameters={
        "type": "object",
        "properties": {
            "group_by": {
                "type": "string",
                "description": "Dimension to group by: "
                + ", ".join(STATS_GROUP_BY_FIELDS)
                + " (default: tool). An unrecognised value falls back to tool.",
            },
            "tool": {
                "type": "string",
                "description": "Restrict to records for this tool before grouping.",
            },
            "type": {
                "type": "string",
                "description": "Restrict to this record type before grouping: "
                "model, tool_call, tool_result.",
            },
            "status": {
                "type": "string",
                "description": "Restrict to this status before grouping: ok or error.",
            },
            "kind": {
                "type": "string",
                "description": "Restrict to this agent kind before grouping: "
                "primary or subagent.",
            },
            "from_record": {"type": "integer", "description": "Lowest record number."},
            "to_record": {"type": "integer", "description": "Highest record number."},
        },
        "required": [],
    },
)

RUN_LOG_SLICE_TOOL_SCHEMA = ToolSchema(
    id="runtime:run_log_slice",
    name=RUN_LOG_SLICE_TOOL_NAME,
    description=(
        "Retrieve a contiguous range of this run's own log records as one "
        "coherent unit, so you can reconstruct a stretch of your own "
        "reasoning instead of assembling it from separate search_run_log "
        "hits. Bounded the same way search_run_log bounds its own output: "
        f"at most {MAX_SLICE_RECORDS} records per call regardless of how "
        "wide the requested range is, and each record's content is "
        "windowed at this run's own tool-result limit. When the requested "
        "range is wider than what one call returns, the rendering states "
        "how many records were shown and the from_record to pass next to "
        "continue."
    ),
    parameters={
        "type": "object",
        "properties": {
            "from_record": {
                "type": "integer",
                "description": "Lowest record number to include. Coerced to "
                "1 if missing or invalid.",
            },
            "to_record": {
                "type": "integer",
                "description": "Highest record number to include. Omit for "
                "a default-width window starting at from_record.",
            },
        },
        "required": ["from_record"],
    },
)


class ToolProvider(Protocol):
    """The capability interface providers implement."""

    def list_catalog(self) -> list[ToolCatalogEntry]: ...

    def load_schema(self, tool_id: str) -> ToolSchema: ...

    def invoke(self, tool_id: str, args: dict) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class ToolPathTarget:
    """One provider-validated path relevant to instruction discovery."""

    path: Path | None
    kind: Literal["exact", "directory", "repository", "outside"]


@runtime_checkable
class PathAwareToolProvider(Protocol):
    """Optional structural path mapping implemented by local file providers."""

    def path_targets(
        self, tool_id: str, args: Mapping[str, Any]
    ) -> tuple[ToolPathTarget, ...]: ...


@dataclass(frozen=True, slots=True)
class _ToolOwnerRecord:
    tool_id: str
    provider: ToolProvider
    source: str | None


@dataclass(frozen=True, slots=True)
class _CatalogSnapshot:
    by_id: Mapping[str, _ToolOwnerRecord]
    by_name: Mapping[str, _ToolOwnerRecord]
    entries: tuple[ToolCatalogEntry, ...]


def build_builtin_gate(*args: Any, **kwargs: Any) -> Any:
    """Thin, monkeypatchable indirection to the real gate builder.

    Defined here (rather than imported at module scope) so this module
    stays dependency-light: `builtin_tool_gate` (which pulls in the MCP
    permission store) is only imported the first time this is actually
    *called*, not merely when `tool_catalog` itself is imported. Keeping
    it as a real module-level name -- instead of a `from ... import` done
    inline inside `_resolve_gate` -- is what lets tests monkeypatch
    `tool_catalog.build_builtin_gate` directly to prove a bare
    `BuiltinToolProvider()` is gated by default (Constraint 6): a
    function-local import only ever binds a local name, never a module
    attribute, so `monkeypatch.setattr(module, "build_builtin_gate", ...)`
    would have nothing to patch without this indirection.
    """
    from tldw_chatbook.Agents.builtin_tool_gate import (
        build_builtin_gate as _build_builtin_gate,
    )

    return _build_builtin_gate(*args, **kwargs)


class GateableTool(NamedTuple):
    """A built-in tool that a ``[tools]`` config flag turns on or off.

    Attributes:
        gate_key: The ``[tools]`` key that enables it (default False).
        module_name: Module under ``tldw_chatbook.Tools`` defining it.
        factory_name: Class name to instantiate.
        tool_name: The name the LLM calls it by.
    """

    gate_key: str
    module_name: str
    factory_name: str
    tool_name: str


#: Built-ins registered unconditionally -- no gate, cannot be turned off.
ALWAYS_ON_BUILTIN_NAMES: tuple[str, ...] = ("calculator", "get_current_datetime")

#: THE source of truth for config-gateable built-ins. Both
#: `BuiltinToolProvider.__init__` and the Settings UI derive from this, so
#: they cannot disagree about which tools exist. The UI needs entries for
#: tools whose gate is OFF -- which is exactly why it cannot ask a provider,
#: since a provider only lists what its gates already permit.
_GATEABLE_BUILTINS: tuple[GateableTool, ...] = (
    GateableTool(
        "read_file_enabled", "file_operation_tools", "ReadFileTool", "read_file"
    ),
    GateableTool(
        "list_directory_enabled",
        "file_operation_tools",
        "ListDirectoryTool",
        "list_directory",
    ),
    GateableTool(
        "write_file_enabled", "file_operation_tools", "WriteFileTool", "write_file"
    ),
    GateableTool(
        "create_note_enabled", "note_management_tools", "CreateNoteTool", "create_note"
    ),
    GateableTool(
        "update_note_enabled", "note_management_tools", "UpdateNoteTool", "update_note"
    ),
    GateableTool(
        "glob_files_enabled", "file_operation_tools", "GlobFiles", "glob_files"
    ),
    GateableTool(
        "grep_files_enabled", "file_operation_tools", "GrepFiles", "grep_files"
    ),
    GateableTool(
        "expand_document_enabled",
        "document_expansion_tool",
        "ExpandDocumentTool",
        "expand_document",
    ),
)


def gateable_builtin_tools() -> tuple[GateableTool, ...]:
    """Every config-gateable built-in, whether or not its gate is on.

    Returns:
        The full table, in registration order.
    """
    return _GATEABLE_BUILTINS


def build_gateable_tool(entry: GateableTool) -> Any:
    """Instantiate ``entry``'s tool class.

    Raises rather than returning ``None`` so callers can report *why* a tool
    is unavailable -- the registration loop logs the exception, and the
    Settings UI degrades the row.

    Args:
        entry: The table entry to construct.

    Returns:
        The instantiated ``Tool``.

    Raises:
        Exception: Whatever import or construction raised.
    """
    import importlib

    module = importlib.import_module(
        f"..Tools.{entry.module_name}", package=__package__
    )
    return getattr(module, entry.factory_name)()


_FILE_AUTHORITY_BUILTIN_NAMES = frozenset(
    {
        "read_file",
        "write_file",
        "list_directory",
        "glob_files",
        "grep_files",
        "expand_document",
    }
)


def redact_root_locator(value: Any, root: Path | None) -> Any:
    """Replace an opaque private-root locator with model-safe relative text.

    Tool-provider results are copied into both model history and run logs.
    Console scratch roots are process-local capabilities, so their absolute
    locator must be removed at that shared boundary. Containers are rebuilt
    recursively because built-in tools return nested JSON-shaped values.

    Args:
        value: Tool result value or error text to sanitize.
        root: Opaque root whose locator must not leave the provider.

    Returns:
        A value of the same JSON-compatible shape with root-owned paths made
        relative and exact root occurrences replaced by ``.``. Non-Console
        callers pass ``None`` and retain their existing output byte-for-byte.
    """
    if root is None:
        return value
    if isinstance(value, str):
        locators = {str(root), root.as_posix()}
        for locator in sorted(locators, key=len, reverse=True):
            if locator:
                value = value.replace(f"{locator}/", "")
                value = value.replace(f"{locator}\\", "")
                value = value.replace(locator, ".")
        return value
    if isinstance(value, dict):
        return {
            key: redact_root_locator(item, root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_root_locator(item, root) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_root_locator(item, root) for item in value)
    return value


class BuiltinToolProvider:
    """Wraps tool_executor's built-in tools behind the provider interface."""

    SOURCE = "builtin"

    def __init__(
        self,
        gate: Any | None = None,
        workspace_id: str | None = None,
        ephemeral: bool = False,
        diff_sink: Callable[[tuple[str, str, str, str]], None] | None = None,
        instruction_root: Path | None = None,
        sandbox_root: Path | None = None,
        sandbox_lease: Callable[[], ContextManager[Path]] | None = None,
    ) -> None:
        # settings-workspaces-folder-roots spec §3: the run's workspace,
        # bound around every tool execution (see `invoke`) so file tools
        # resolve THIS run's folder roots -- never whatever workspace the
        # user happens to be looking at when the tool actually fires.
        # `None` (the default -- every construction site that never cares
        # about workspace-scoped file roots, e.g. Settings-time enumeration
        # in `builtin_tool_gate.builtin_permission_rows`) leaves
        # `allowed_file_roots` to fall back to the active workspace.
        self._workspace_id = workspace_id
        self._sandbox_root = (
            Path(sandbox_root).resolve() if sandbox_root is not None else None
        )
        self._sandbox_lease = sandbox_lease
        self._instruction_root = (
            Path(instruction_root).resolve() if instruction_root is not None else None
        )
        # final-review F4: whether THIS run's owning Console session is
        # temporary. Mirrors `_workspace_id` exactly -- `False` (the
        # default) preserves every pre-existing construction site's
        # behavior unchanged; `console_agent_bridge._compose_run_registry_
        # and_allowed` threads the real value through for an actual
        # Console run. `invoke()` uses it to refuse the write-shaped
        # built-ins (`create_note`/`update_note`/`write_file`) -- an
        # ordinary agent reply composes and dispatches these exactly like
        # any other built-in, independently of the Console UI action-id
        # registry in `Chat/console_ephemeral.py`.
        self._ephemeral = ephemeral
        self._tools = {t.name: t for t in (CalculatorTool(), DateTimeTool())}
        # task-584: surface the app's existing sandbox-rooted file tools to the
        # agent loop. They were registered on the global ToolExecutor but never
        # reachable from here, so retained script output -- deliberately written
        # under the file-tool sandbox root -- had no consumer. Behind the SAME
        # [tools] gates that already govern them, which default to DISABLED:
        # this changes reachability, not the default posture. TASK-545 P2 adds
        # the mutating tools on the same terms.
        for entry in _GATEABLE_BUILTINS:
            try:
                from ..config import coerce_bool_setting, get_cli_setting

                # task-3240 Critical prerequisite: get_cli_setting returns the
                # RAW TOML value, and a mis-typed quoted "false" is truthy --
                # raw truthiness would have REGISTERED the tool while a
                # coerced UI (the MCP-hub gate affordance) showed it OFF.
                # coerce_bool_setting applies load_settings' own bool rules
                # ("false"/unrecognized -> False), matching every other
                # reader of a [tools]/[console] gate (see
                # Agents/local_tool_provider.py's web_deep_search gate and
                # Agents/builtin_tool_gate.py's all_tool_gates()).
                if not coerce_bool_setting(
                    get_cli_setting("tools", entry.gate_key, False), False
                ):
                    continue
                tool = build_gateable_tool(entry)
            except Exception as exc:  # noqa: BLE001 — an unavailable tool is just absent
                # Log rather than vanish silently. The gate-off path `continue`s
                # ABOVE this handler, so reaching here means the user asked for
                # the tool and it could not be built -- indistinguishable from
                # "gate is off" without this line. That is not hypothetical:
                # note_management_tools was unimportable on dev for an unknown
                # period (it imported a name that exists only inside a string
                # literal in config.py) and nothing surfaced it. The legacy
                # path logged the same failure before it was retired (P3).
                # opt(exception=True), not exc_info=True: loguru ignores the
                # latter, and an import-time failure is undiagnosable without
                # the traceback naming the module and line.
                logger.opt(exception=True).warning(
                    f"Could not register builtin tool {entry.factory_name} "
                    f"(gate {entry.gate_key} is enabled): {exc}"
                )
                continue
            self._tools[tool.name] = tool
        # `None` means "build the real gate on first use" -- NOT "ungated".
        # Every construction site (console_agent_bridge's default registry
        # and its per-run registry) passes nothing today, so an ungated
        # default would silently leave the shipping path unprotected.
        self._gate = gate
        # TASK-1366: optional UI-side channel for raw before/after file
        # contents, invoked at the strip seam in `invoke()` with a single
        # ``(tool_name, file_path, old_content, new_content)`` tuple just
        # BEFORE the raw keys are removed from the LLM/run-log-bound dict.
        # This is the ONLY way the live Console can render a diff: the
        # stripped JSON text is all that leaves this method. `None` (the
        # default) means no UI is listening -- behavior is byte-identical
        # to pre-diff-channel runs. The sink runs on the tool call's
        # PER-CALL DAEMON THREAD (AgentService._call_with_timeout) -- on
        # timeout/cancel that thread is abandoned unjoined and the sink
        # can fire LATE, after the call's result step already passed (the
        # bridge's pairing tolerates this; see console_agent_bridge.
        # _pair_step_diff). Its exceptions are swallowed (a UI failure
        # must never break a tool call), so it must be cheap, non-
        # blocking, and safe to call cross-thread -- the bridge hands in
        # a `deque.append` (single-argument contract, atomic in CPython).
        self._diff_sink = diff_sink

    @property
    def sandbox_root(self) -> Path | None:
        """Return this provider's explicit run sandbox, when one was bound.

        Returns:
            The resolved per-run sandbox root, or ``None`` when absent.
        """
        return self._sandbox_root

    @property
    def sandbox_lease(self) -> Callable[[], ContextManager[Path]] | None:
        """Return the lease factory paired with the explicit run sandbox.

        Returns:
            A context-manager factory that leases the sandbox generation, or
            ``None`` when the provider has no explicit sandbox authority.
        """
        return self._sandbox_lease

    @contextmanager
    def _file_authority(self) -> Iterator[None]:
        """Keep the explicit scratch generation alive for one file access."""
        from tldw_chatbook.Tools.workspace_file_roots import run_file_sandbox

        lease = (
            self._sandbox_lease() if self._sandbox_lease is not None else nullcontext()
        )
        sandbox = (
            run_file_sandbox(self._sandbox_root)
            if self._sandbox_root is not None
            else nullcontext()
        )
        with lease, sandbox:
            yield

    def _tool_id(self, name: str) -> str:
        return f"{self.SOURCE}:{name}"

    def tool_for(self, name: str) -> Any | None:
        """Return the built-in ``Tool`` registered under ``name``, if any."""
        return self._tools.get(name)

    def timeout_for(self, tool_id: str) -> float | None:
        """Return this tool's own timeout ceiling, if it declares one."""
        tool = self._tools.get(tool_id.split(":", 1)[-1])
        seconds = float(getattr(tool, "timeout_seconds", 0.0) or 0.0)
        return seconds if seconds > 0 else None

    def path_targets(
        self, tool_id: str, args: Mapping[str, Any]
    ) -> tuple[ToolPathTarget, ...]:
        """Map enabled built-in file tools to their validated target path."""
        name = tool_id.split(":", 1)[-1]
        root = self._instruction_root
        arguments = {
            "read_file": ("file_path", "exact", False),
            "write_file": ("file_path", "exact", True),
            "list_directory": ("directory_path", "directory", False),
        }
        mapping = arguments.get(name)
        if root is None or mapping is None or name not in self._tools:
            return ()
        argument, kind, write = mapping
        value = args.get(argument)
        if not isinstance(value, (str, Path)):
            return ()

        from tldw_chatbook.Tools.file_operation_tools import (
            _tool_sandbox_root,
            allowed_file_roots,
        )
        from tldw_chatbook.Tools.workspace_file_roots import run_workspace
        from tldw_chatbook.Utils.path_validation import validate_path_multi

        with self._file_authority(), run_workspace(self._workspace_id):
            roots = allowed_file_roots(write=write, sandbox_root=_tool_sandbox_root())
            path = validate_path_multi(value, roots)
            try:
                path.relative_to(root)
            except ValueError:
                return (ToolPathTarget(path=path, kind="outside"),)
            return (ToolPathTarget(path=path, kind=kind),)

    def _resolve_gate(self) -> Any:
        """Return the provider's gate, building one lazily on first use.

        Note: nothing here calls `begin_turn()` on a lazily-built gate,
        so its permission-store payload is loaded once and never
        invalidated for the life of this provider. Harmless today --
        `build_builtin_gate()`'s default, service-less gate always has
        an empty `{}` payload -- but any gate handed to (or built by)
        this provider must be driven by a caller that calls
        `begin_turn()` once per turn, as `console_chat_controller`'s
        review-hook path already does, or its permission state will
        freeze at first use.
        """
        if self._gate is None:
            # Module-global lookup (not a local import) so a test's
            # monkeypatch of `tool_catalog.build_builtin_gate` is honored.
            self._gate = build_builtin_gate()
        return self._gate

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(
                id=self._tool_id(t.name),
                name=t.name,
                one_line_description=t.description,
                source=self.SOURCE,
            )
            for t in self._tools.values()
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        name = tool_id.split(":", 1)[1]
        tool = self._tools[name]
        return ToolSchema(
            id=tool_id,
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
        )

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        name = tool_id.split(":", 1)[1]
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"Unknown builtin tool: {name}")
        # final-review F4: a temporary Console session must refuse the
        # write-shaped built-ins outright, BEFORE the approval gate below
        # -- this is an absolute local-durability boundary, not a
        # permission decision, so it must win even for a tool the user has
        # already approved (session or always-allow). Checked first, same
        # reasoning `_console_save_as_destinations` already uses for the
        # per-message Save-as row: "the write itself is the problem,
        # service/approval readiness is moot."
        if self._ephemeral:
            from tldw_chatbook.Chat.console_ephemeral import blocked_reason

            reason = blocked_reason(name, ephemeral=True)
            if reason is not None:
                return ToolResult.blocked(reason)
        # Defense in depth: the run-level review hook is the primary gate
        # (it batches approvals into one card per turn), but a caller that
        # reaches invoke() without going through it must still not execute
        # ungated. A gate that raises fails CLOSED -- never into the pure
        # loop, which must not see exceptions from tool invocation.
        # PR2a Task 5: the gate keys this turn's stamps by run, and only
        # the DISPATCHING run's own stamp may permit this call. The
        # `ToolProvider.invoke` Protocol has no run parameter, so the run
        # id rides `run_context` (bound by `AgentService` around each
        # invocation -- see that module's docstring for why). Outside any
        # run this resolves to `""`, which matches no stamp a review hook
        # ever writes, so such a call falls through to the resolved
        # permission state exactly as it did before per-run keying.
        try:
            refusal = self._resolve_gate().check(tool, current_run_id())
        except Exception as exc:  # noqa: BLE001 — fail closed
            return ToolResult(ok=False, error=f"permission check failed: {exc}")
        if refusal is not None:
            return ToolResult.blocked(refusal)
        from tldw_chatbook.Tools.workspace_file_roots import run_workspace

        authority = (
            self._file_authority()
            if name in _FILE_AUTHORITY_BUILTIN_NAMES
            else nullcontext()
        )
        try:
            # Providers bridge async tools; the loop's interface is sync.
            # Safe here: the service runs in a worker thread with no
            # running event loop. `run_workspace` binds this run's
            # workspace for the DURATION of the call only (context-managed,
            # reset in its own `finally`) so file tools (`allowed_file_
            # roots`) resolve THIS run's folder bindings, never a stale or
            # concurrent run's. `self._workspace_id=None` keeps the
            # ContextVar at `None`, which is `allowed_file_roots`' own
            # documented fallback to the active workspace.
            with authority, run_workspace(self._workspace_id):
                raw = asyncio.run(tool.execute(**args))
        except Exception as exc:  # noqa: BLE001 — captured, never escapes
            return ToolResult(
                ok=False,
                error=redact_root_locator(str(exc), self._sandbox_root),
            )
        if isinstance(raw, dict) and raw.get("error"):
            return ToolResult(
                ok=False,
                error=redact_root_locator(str(raw["error"]), self._sandbox_root),
            )
        if isinstance(raw, dict):
            # Raw before/after contents captured for UI diff rendering
            # (TASK-1351) are live-session display state only. This is the
            # seam where a builtin tool's result dict becomes the JSON text
            # that feeds BOTH the model history (_append_tool_result) and
            # the on-disk run log (_emit_record) -- strip here so the raw
            # contents are never replayed to a provider or persisted.
            from tldw_chatbook.Tools.file_operation_tools import DIFF_CONTENT_KEYS

            # TASK-1366: hand the raw contents to the UI diff channel
            # FIRST, while they still exist -- after the strip below they
            # are gone from everything this method returns. Only the sink
            # (an in-memory, live-session channel) ever sees them.
            if self._diff_sink is not None:
                old_content = raw.get("old_content")
                new_content = raw.get("new_content")
                if isinstance(old_content, str) and isinstance(new_content, str):
                    try:
                        self._diff_sink(
                            (
                                name,
                                redact_root_locator(
                                    str(raw.get("file_path") or "file"),
                                    self._sandbox_root,
                                ),
                                old_content,
                                new_content,
                            )
                        )
                    except Exception:  # noqa: BLE001 — UI failure never breaks a tool call
                        logger.opt(exception=True).warning(
                            "diff_sink raised during tool result capture; "
                            "console diff rows will be missing for this write"
                        )
            raw = {
                key: value for key, value in raw.items() if key not in DIFF_CONTENT_KEYS
            }
        raw = redact_root_locator(raw, self._sandbox_root)
        content = json.dumps(raw) if isinstance(raw, (dict, list)) else str(raw)
        return ToolResult(ok=True, content=content)


def intersect_skill_tools(
    skill_allowed_tools: list[str] | None,
    builtin_names: Iterable[str],
) -> tuple[str, ...]:
    """A skill's `allowed_tools` narrows the given tool set; never grants.

    ``None`` means the skill did not narrow — the whole set passes through.
    Otherwise only names present in both survive, ordered by
    ``builtin_names`` (not the skill's own order) so callers get a stable,
    registry-consistent ordering regardless of how the skill listed them.

    Despite the parameter name (kept for call-site compatibility), the second
    argument is the full narrowing set: since phase 3c the bridge passes
    builtins + local tool names, so a skill may narrow against both — never
    against skill, runtime, or MCP names.

    Args:
        skill_allowed_tools: The skill's own declared ``allowed_tools``
            list (front-matter), or ``None`` when the skill did not narrow
            its child's tool set at all.
        builtin_names: The narrowing set, in registry order — the widest
            set a narrowed list can ever be intersected down to; a name
            absent here can never be granted regardless of what the skill
            declares.

    Returns:
        ``tuple(builtin_names)`` unchanged when ``skill_allowed_tools`` is
        ``None``; otherwise the subset of ``builtin_names`` also present in
        ``skill_allowed_tools``, preserving ``builtin_names``' order.
    """
    if skill_allowed_tools is None:
        return tuple(builtin_names)
    allowed = set(skill_allowed_tools)
    return tuple(name for name in builtin_names if name in allowed)


class SkillToolProvider:
    """Exposes trusted, model-invocable skills as catalog tools.

    Built from a per-run snapshot of skill summaries (plain mappings with
    "name", "description", "argument_hint") — never imports Skills_Interop
    itself, so this module stays importable without that subsystem and the
    catalog is always as fresh as the snapshot the caller passed in (the
    per-run freshness doctrine: callers re-read skills at run start, not
    once at import time).

    ``invoke()`` deliberately raises: skill tools never execute via plain
    provider.invoke(). They route through the run-scoped spawn executor
    (budget-counted, cancellable, DB-lineage-tracked sub-agent runs — see
    the skills design doc's Architecture section). This method exists only
    to satisfy the ToolProvider protocol; calling it directly is a bug.
    """

    SOURCE = "skill"

    def __init__(self, entries: list[Mapping]) -> None:
        self._entries = list(entries)

    def _tool_id(self, name: str) -> str:
        return f"{self.SOURCE}:{name}"

    def list_catalog(self) -> list[ToolCatalogEntry]:
        """Return one cheap-to-list catalog row per skill entry.

        Returns:
            A `ToolCatalogEntry` for each skill this provider was built
            with, in the order the entries were passed to `__init__`; each
            entry's `id` is `"skill:<name>"` and its `source` is
            `SOURCE` (`"skill"`).
        """
        return [
            ToolCatalogEntry(
                id=self._tool_id(e["name"]),
                name=e["name"],
                one_line_description=e["description"],
                source=self.SOURCE,
            )
            for e in self._entries
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        """Return the full tool schema for one previously-listed skill entry.

        Args:
            tool_id: A catalog id previously returned by `list_catalog`
                (``"skill:<name>"``).

        Returns:
            A `ToolSchema` whose single parameter is a free-form ``args``
            string (described by the skill's own ``argument_hint``, or its
            ``description`` when no hint was given) — skills never expose a
            structured parameter schema the way builtin tools do.

        Raises:
            StopIteration: ``tool_id``'s name does not match any entry this
                provider was built with (mirrors `next()`'s own behavior
                with no default; never expected in practice since
                `tool_id` always comes from this provider's own
                `list_catalog`).
        """
        name = tool_id.split(":", 1)[1]
        entry = next(e for e in self._entries if e["name"] == name)
        hint = entry.get("argument_hint") or entry["description"]
        return ToolSchema(
            id=tool_id,
            name=name,
            description=entry["description"],
            parameters={
                "type": "object",
                "properties": {"args": {"type": "string", "description": hint}},
                "required": [],
            },
        )

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        """Never called: satisfies the `ToolProvider` protocol only.

        Args:
            tool_id: Unused — present only to match the protocol shape.
            args: Unused — present only to match the protocol shape.

        Raises:
            RuntimeError: Always. A skill-tool call must route through the
                run-scoped spawn executor (see this class's own docstring
                and `console_agent_bridge._BridgeSkillRunner`), never
                through a plain `ToolProvider.invoke`. Reaching this method
                at all is a caller bug.
        """
        raise RuntimeError(
            "SkillToolProvider.invoke must not be called; skills route "
            "through the run-scoped spawn executor"
        )


class ToolCatalogRegistry:
    """Ordered provider registry: catalog, search, schema, invocation."""

    def __init__(self, *, ephemeral: bool = False) -> None:
        self._providers: list[ToolProvider] = []
        self._builtin_library_provider: ToolProvider | None = None
        self._builtin_library_authority: BuiltinLibraryAuthority | None = None
        # Whether the Console session owning THIS run is temporary ("not
        # saved locally"). Enforced in `invoke_by_name` -- the one choke
        # point every provider's `invoke()` is reached through -- rather
        # than inside any individual provider, so the guarantee does not
        # depend on each provider (including ones added later) remembering
        # to implement it. `BuiltinToolProvider` keeps its own equivalent
        # check as defense in depth; this one is the load-bearing gate.
        # `False` (the default) preserves every pre-existing construction
        # site's behavior exactly.
        self._ephemeral = ephemeral
        # One immutable owner snapshot is built lazily per run. Public
        # lookups retain the returned object, so a concurrent reset or
        # registration cannot make one lookup reread a different generation.
        self._catalog_snapshot: _CatalogSnapshot | None = None
        self._catalog_lock = threading.RLock()
        self._catalog_generation = 0
        # Workspace assistant defaults (Task 7): THIS run's persona-policy
        # call caps, enforced in `invoke_by_name` below. Set only by the
        # per-run composition (`console_agent_bridge.
        # _compose_run_registry_and_allowed`) on a FRESHLY built registry;
        # deliberately NOT cleared by `reset_catalog_cache`, which
        # `AgentService` calls at the top of the run tree -- AFTER this
        # registry was composed and its policy armed, and BEFORE dispatch.
        # Clearing there would disarm every run's caps.
        self._run_tool_policy: RunToolPolicy | None = None

    def set_run_tool_policy(self, policy: RunToolPolicy | None) -> None:
        """Arm (or clear) this registry's per-run persona-policy call caps.

        Workspace assistant defaults (Task 7): the per-run composition
        builds a fresh ``RunToolPolicy`` from the workspace persona's
        ``max_calls_per_turn`` rule verdicts and arms it here, so the cap
        binds at ``invoke_by_name`` -- the one choke point every provider's
        ``invoke()`` is reached through. ``None`` (the default, and the
        no-rules posture) leaves invocation behavior unchanged.

        Args:
            policy: The policy to consult before every dispatch, or ``None``
                to disable cap enforcement on this registry.
        """
        self._run_tool_policy = policy

    def register_provider(self, provider: ToolProvider) -> None:
        # The append lives inside the lock too, alongside the three
        # invalidations: a concurrent reader must never be able to observe
        # the new provider in `self._providers` while still holding a
        # cache built before it was appended (or vice versa).
        with self._catalog_lock:
            self._providers.append(provider)
            # A newly registered provider's tools aren't reflected in any
            # cache already built — invalidate so the next lookup rebuilds it.
            self._catalog_snapshot = None
            self._catalog_generation += 1

    def register_builtin_library_provider(
        self,
        provider: ToolProvider,
        authority: BuiltinLibraryAuthority | None,
    ) -> bool:
        """Register one exact in-tree Library provider with its live capability.

        Source strings and structural lookalikes are deliberately irrelevant:
        only the concrete built-in provider classes and the exact authority
        object currently issued by that same instance cross this boundary.
        """
        provider_type = type(provider)
        if provider_type is LibraryToolProvider:
            expected_names = frozenset(LIBRARY_TOOL_DESCRIPTORS)
        elif provider_type is LibraryRagToolProvider:
            expected_names = frozenset({RAG_TOOL_NAME})
        else:
            return False
        if (
            not isinstance(authority, BuiltinLibraryAuthority)
            or authority.assistant_access is not ConsoleAssistantLibraryAccess.ALLOWED
            or authority.reserved_names is not LIBRARY_RESERVED_TOOL_NAMES
            or not provider.authenticates_builtin_authority(authority)
        ):
            return False
        try:
            entries = provider.list_catalog()
        except Exception:  # noqa: BLE001 - malformed provider fails closed
            return False
        if (
            frozenset(entry.name for entry in entries) != expected_names
            or any(entry.source != "library" for entry in entries)
        ):
            return False
        with self._catalog_lock:
            if self._builtin_library_provider is not None:
                return False
            self._builtin_library_provider = provider
            self._builtin_library_authority = authority
            self._providers.append(provider)
            self._catalog_snapshot = None
            self._catalog_generation += 1
        return True

    def _authenticated_builtin_library_name(
        self, provider: ToolProvider, name: str
    ) -> bool:
        """Return whether ``name`` is live-authorized for this exact provider."""
        authority = self._builtin_library_authority
        return bool(
            provider is self._builtin_library_provider
            and isinstance(authority, BuiltinLibraryAuthority)
            and authority.assistant_access is ConsoleAssistantLibraryAccess.ALLOWED
            and authority.reserved_names is LIBRARY_RESERVED_TOOL_NAMES
            and name in LIBRARY_RESERVED_TOOL_NAMES
            and provider.authenticates_builtin_authority(authority)
        )

    def reset_catalog_cache(self) -> None:
        """Drop the owner-map/name-map cache; call once at the start of a run.

        Cache scope is PER RUN: the catalog is listed fresh at run start
        (``AgentService.run_turn`` calls this before dispatching), so any
        skill CRUD (or other provider mutation) between runs is always
        picked up. No cross-run invalidation signal is needed beyond this
        single reset — see the skills spec's Catalog scale section.
        """
        with self._catalog_lock:
            self._catalog_snapshot = None
            self._catalog_generation += 1

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return list(self._ensure_catalog_cache().entries)

    def find(
        self,
        query: str,
        *,
        allowed_names: Iterable[str] | None = None,
        limit: int = FIND_TOOLS_RESULT_LIMIT,
    ) -> list[ToolCatalogEntry]:
        """Return deterministic, relevance-ranked catalog metadata.

        Args:
            query: Case-insensitive name or description substring to find.
            allowed_names: Optional name allow-list applied before ranking.
            limit: Maximum number of matching catalog rows to return.

        Returns:
            Matching entries ordered by exact, prefix, name-substring, then
            description-substring relevance with deterministic tie-breaking.
        """
        needle = query.strip().casefold()
        if not needle:
            return []
        allowed = None if allowed_names is None else frozenset(allowed_names)
        ranked: list[tuple[int, str, str, ToolCatalogEntry]] = []
        for entry in self.list_catalog():
            if allowed is not None and entry.name not in allowed:
                continue
            name = entry.name.casefold()
            description = entry.one_line_description.casefold()
            if name == needle:
                rank = 0
            elif name.startswith(needle):
                rank = 1
            elif needle in name:
                rank = 2
            elif needle in description:
                rank = 3
            else:
                continue
            ranked.append((rank, name, entry.id, entry))
        ranked.sort(key=lambda item: item[:3])
        return [item[3] for item in ranked[: max(int(limit), 0)]]

    def _build_owner_cache(
        self,
    ) -> _CatalogSnapshot:
        by_id: dict[str, _ToolOwnerRecord] = {}
        by_name: dict[str, _ToolOwnerRecord] = {}
        accepted_entries: list[ToolCatalogEntry] = []
        for provider in self._providers:
            for entry in provider.list_catalog():
                if (
                    self._ephemeral
                    and entry.source == "library"
                    and not self._authenticated_builtin_library_name(
                        provider, entry.name
                    )
                ):
                    continue
                if entry.id in by_id or entry.name in by_name:
                    continue
                record = _ToolOwnerRecord(
                    tool_id=entry.id,
                    provider=provider,
                    source=entry.source,
                )
                by_id[entry.id] = record
                by_name[entry.name] = record
                accepted_entries.append(entry)
        return _CatalogSnapshot(
            by_id=MappingProxyType(by_id),
            by_name=MappingProxyType(by_name),
            entries=tuple(accepted_entries),
        )

    def _ensure_catalog_cache(self) -> _CatalogSnapshot:
        # This is the fix MCP (task-201) also needs: a network-backed
        # provider must not re-list_catalog() per lookup. The ID and name
        # indexes are built together from one provider sweep and published as
        # one snapshot. Task-327's per-call daemon threads can outlive a run;
        # the lock and generation check keep such a build from publishing
        # after reset_catalog_cache() has invalidated its generation.
        with self._catalog_lock:
            for _attempt in range(2):
                if self._catalog_snapshot is not None:
                    return self._catalog_snapshot
                generation = self._catalog_generation
                built = self._build_owner_cache()
                if generation != self._catalog_generation:
                    continue
                self._catalog_snapshot = built
                return built
            raise RuntimeError("tool catalog changed during cache build")

    def _owner_and_id(self, tool_id: str):
        record = self._ensure_catalog_cache().by_id.get(tool_id)
        return record.provider if record is not None else None

    def _source_for(self, tool_id: str) -> str | None:
        """Return the catalog ``source`` that owns ``tool_id``, if known.

        ``None`` when the id is absent from the cache — which the ephemeral
        gate treats as an unaudited source and refuses, never as "allow".
        """
        record = self._ensure_catalog_cache().by_id.get(tool_id)
        return record.source if record is not None else None

    def load_schema(self, tool_id: str) -> ToolSchema:
        provider = self._owner_and_id(tool_id)
        if provider is None:
            raise KeyError(f"Unknown tool id: {tool_id}")
        return provider.load_schema(tool_id)

    def resolve_name(self, name: str) -> str | None:
        record = self._ensure_catalog_cache().by_name.get(name)
        return record.tool_id if record is not None else None

    def _owner_record_for_name(self, name: str) -> _ToolOwnerRecord | None:
        return self._ensure_catalog_cache().by_name.get(name)

    def resolve_owner_for_name(self, name: str) -> tuple[str, ToolProvider] | None:
        """Atomically resolve one LLM-facing name to its cached first owner."""
        record = self._owner_record_for_name(name)
        if record is None:
            return None
        return record.tool_id, record.provider

    def invoke_by_name(self, name: str, args: dict) -> ToolResult:
        record = self._owner_record_for_name(name)
        if record is None:
            return ToolResult(ok=False, error=f"Unknown tool: {name}")
        tool_id, provider = record.tool_id, record.provider
        # Workspace assistant defaults (Task 7): persona-policy call caps,
        # refused BEFORE dispatch in the exact error-`ToolResult` shape the
        # unknown-tool branch above uses. Narrowing-only -- a capped tool is
        # refused, never widened; the run id keys the counters so concurrent
        # sub-agent runs sharing this registry keep independent budgets.
        if self._run_tool_policy is not None:
            allowed, refusal = self._run_tool_policy.check(current_run_id(), name)
            if not allowed:
                return ToolResult(ok=False, error=refusal)
        # THE choke point for the temporary-session ("not saved locally")
        # guarantee. Every provider's invoke() is reached through this one
        # line, so gating here -- rather than in each provider -- is what
        # makes the guarantee hold for MCP tools, skill tools, and any
        # provider added later, without each of them having to opt in.
        # `tool_blocked_reason` owns the policy (built-ins judged per name,
        # everything else refused); an unresolvable source refuses too.
        # Returns a ToolResult rather than raising: the pure loop must never
        # see an exception out of tool invocation.
        if self._ephemeral:
            if self._authenticated_builtin_library_name(provider, name):
                return provider.invoke(tool_id, args)
            from tldw_chatbook.Chat.console_ephemeral import tool_blocked_reason

            reason = tool_blocked_reason(name, source=record.source, ephemeral=True)
            if reason is not None:
                return ToolResult.blocked(reason)
        return provider.invoke(tool_id, args)

    def timeout_for(self, name: str) -> float | None:
        """Resolve a tool's per-call timeout override by LLM-facing name.

        Duck-typed like the rest of the provider interface: a provider that
        does not implement ``timeout_for`` simply has no overrides, so MCP
        and skill tools keep using the run budget unchanged.

        Args:
            name: The tool name the model called.

        Returns:
            A positive seconds value, or None to use the run default.
        """
        record = self._owner_record_for_name(name)
        if record is None:
            return None
        getter = getattr(record.provider, "timeout_for", None)
        return getter(record.tool_id) if getter is not None else None

    def execution_policy_for(self, name: str) -> ToolExecutionPolicy:
        """Resolve code-owned execution ownership for one tool name.

        Providers without this optional capability, missing tools, invalid
        values, and provider errors retain the bounded abandonable behavior.
        Only an explicit enum value may disable the runtime timeout.
        """
        record = self._owner_record_for_name(name)
        if record is None:
            return ToolExecutionPolicy.BOUNDED_ABANDONABLE
        getter = getattr(record.provider, "execution_policy_for", None)
        if getter is None:
            return ToolExecutionPolicy.BOUNDED_ABANDONABLE
        try:
            policy = getter(record.tool_id)
        except Exception:  # noqa: BLE001 - unknown policy fails closed
            return ToolExecutionPolicy.BOUNDED_ABANDONABLE
        return (
            policy
            if isinstance(policy, ToolExecutionPolicy)
            else ToolExecutionPolicy.BOUNDED_ABANDONABLE
        )


def probe_initial_catalog(
    registry: ToolCatalogRegistry,
    allowed_names: Iterable[str],
    max_schema_tokens: int,
    measure_schema_set: Callable[[tuple[ToolSchema, ...]], int],
) -> tuple[ToolSchema, ...] | None:
    """Return every allowed schema only when each cumulative set is proven fit.

    Args:
        registry: Catalog whose allowed schemas are probed in stable order.
        allowed_names: Tool names eligible for initial disclosure.
        max_schema_tokens: Maximum measured size for the full disclosed set.
        measure_schema_set: Callback that measures each cumulative schema set.

    Returns:
        Every allowed schema when all cumulative measurements fit; otherwise
        ``None`` so the caller can switch to progressive discovery.
    """
    if type(max_schema_tokens) is not int or max_schema_tokens <= 0:
        return None
    allowed = frozenset(allowed_names)
    schemas: list[ToolSchema] = []
    try:
        for entry in registry.list_catalog():
            if entry.name not in allowed:
                continue
            schemas.append(registry.load_schema(entry.id))
            measured = measure_schema_set(tuple(schemas))
            if type(measured) is not int or measured <= 0:
                return None
            if measured > max_schema_tokens:
                return None
    except Exception:
        return None
    return tuple(schemas)
