"""Pure projection of a Console conversation into a trajectory snapshot.

Folds persisted message rows, the schema-v38 trajectory sidecar
(``message_trajectory_metadata``), per-message usage, variant sets, and
compaction-attempt records into the turn/record shape the trajectory view
renders (spec: ``Docs/superpowers/specs/
2026-08-14-console-trajectory-view-design.md``, "Projection module").

Purity contract
    This module is stdlib-only plus ``ProviderUsage`` (itself stdlib-only).
    It never imports Textual or the DB layer, and it never queries
    anything: every fact arrives as a plain sequence of dataclasses /
    mappings, so the projection is fully unit-testable and safe to run in
    any worker. Input shapes are duck-typed:

    - ``messages``: ``messages``-table-shaped rows (mappings such as
      ``sqlite3.Row``/dict with ``id``, ``sender``, ``content``,
      ``timestamp``, ``parent_message_id``, ``deleted``) OR
      ``ConsoleChatMessage`` models (joined on
      ``persisted_message_id or id``).
    - ``usage_by_id``: mapping of message id -> ``ProviderUsage | None``.
    - ``traj_rows``: ``TrajectoryRowRead``-shaped objects (attributes
      ``message_id``, ``turn_id``, ``seq``, ``event_kind``,
      ``step_started_at``, ``first_token_at``, ``completed_at``,
      ``model``, ``provider``, ``payload_json``), ordered or not.
    - ``variant_sets``: ``ConsoleVariantSet``-shaped objects
      (``turn_id``, ``variants`` items exposing ``content``,
      ``selected_index``).
    - ``compaction_records``: ``list_auxiliary_attempts``-shaped mappings
      (``purpose``, ``status``, ``started_at``, ``finished_at``,
      ``provider``, ``model``, ``provider_usage_json``).
    - ``agent_runs`` / ``agent_steps``: AgentRunsDB-shaped plain rows.
    - ``retrieval_runs``: safe citation-provenance ``EvidenceRun`` summaries.

Never-fabricate contract
    Timing is surfaced exactly as stored. NULL sidecar timing becomes
    ``None`` fields, and tool rows created at marker-append time keep
    their append-time (zero-duration) stamps verbatim -- the projection
    never derives, defaults, or "corrects" a timestamp.
"""

from __future__ import annotations

import ast
import hashlib
import heapq
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from tldw_chatbook.Chat.provider_usage import ProviderUsage

__all__ = [
    "TrajectoryRecord",
    "TrajectorySnapshot",
    "TrajectoryTurn",
    "contains_local_path",
    "derive_trajectory",
]

#: Cap for ``TrajectoryRecord.content_preview`` (spec: first 120 chars).
PREVIEW_MAX_CHARS = 120

#: Ledger ``kind`` values (spec data model).
KIND_USER = "user"
KIND_SYSTEM = "system"
KIND_ASSISTANT = "assistant"
KIND_TOOL_CALL = "tool_call"
KIND_TOOL_RESULT = "tool_result"
KIND_COMPACTION = "compaction"
KIND_USER_FEEDBACK = "user_feedback"

_TOOL_KINDS = frozenset({KIND_TOOL_CALL, KIND_TOOL_RESULT})
# The preparation disclosure is projected by its owning pure module. It is never
# a message row or nested tool row in the generic trajectory.
_SIDECAR_ONLY_KINDS = frozenset({"library_preparation"})
# Kinds that nest UNDER the message they key on rather than being that
# message's own sidecar row. Feedback (task-17169) is keyed to the message
# it critiques, so treating it as that message's row would displace the
# real one -- taking its timing and turn attribution with it.
_RENDERED_ROLES = frozenset({KIND_USER, KIND_SYSTEM, KIND_ASSISTANT})
_COMPACTION_PURPOSE = "conversation_compaction"

# Sentinel for "no sidecar seq" in sort keys; larger than any real seq.
_NO_SEQ = float("inf")

_LOCAL_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9:/<])(?:~/|/)[^\s\"'<>]+"
    r"|(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/][^\\/\s]+)"
    r"[^\s\"'<>]*"
)
_HTTP_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_HTTP_REQUEST_TARGET_RE = re.compile(
    r"^(\s*(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+)\S+(?=\s+HTTP/\d)",
    re.IGNORECASE,
)
_SAFE_ROUTE_ROOTS = frozenset({"api", "docs", "help", "v1", "v2", "v3"})
_PATH_CONTEXT_KEYS = frozenset({"path", "file_path", "file", "cwd", "directory"})
_STRUCTURED_PATH_FIELD_RE = re.compile(
    r"[\"']?(?:path|file_path|file|cwd|directory)[\"']?\s*[:=]\s*"
    r"[\"']?(?:file://|~/|/|[A-Za-z]:[\\/]|\\\\)",
    re.IGNORECASE,
)
_LOCAL_FILE_COMMANDS = frozenset({"cd", "ls", "cat", "open", "less", "head", "tail"})
_STRUCTURED_PARSE_MAX_CHARS = 4_096
_STRUCTURED_NESTING_MAX = 64
_PYTHON_STACK_RE = re.compile(
    r"\bfile\s+[\"']?[^\"',\s]+[\\/][^\"',]+[\"']?,\s*line\s+\d+",
    re.IGNORECASE,
)
_JS_STACK_RE = re.compile(
    r"^\s*at\s+.*[\\/][^\s)]+:\d+(?::\d+)?\)?\s*$", re.IGNORECASE
)


def contains_local_path(value: str) -> bool:
    """Return whether text contains a local path or file URI."""
    stripped_value = value.strip()
    if stripped_value.startswith(("{", "[", "(")):
        depth = 0
        quote: str | None = None
        escaped = False
        for char in stripped_value:
            if escaped:
                escaped = False
                continue
            if quote is not None:
                if char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in "\"'":
                quote = char
            elif char in "{[(":
                depth += 1
                if depth > _STRUCTURED_NESTING_MAX:
                    return True
            elif char in "}])":
                depth = max(0, depth - 1)
        if len(stripped_value) > _STRUCTURED_PARSE_MAX_CHARS:
            structured = None
            if _STRUCTURED_PATH_FIELD_RE.search(stripped_value):
                return True
        else:
            try:
                if re.match(r"^[{[(]\s*'", stripped_value):
                    structured = ast.literal_eval(stripped_value)
                else:
                    structured = json.loads(stripped_value)
            except (MemoryError, RecursionError, OverflowError):
                return True
            except (SyntaxError, TypeError, ValueError):
                structured = None

        def has_path_context(item: Any) -> bool:
            if isinstance(item, Mapping):
                for key, nested in item.items():
                    normalized = re.sub(r"[\s-]+", "_", str(key).lower())
                    if normalized in _PATH_CONTEXT_KEYS and isinstance(nested, str):
                        candidate = nested.strip()
                        if candidate.lower().startswith("file://") or candidate.startswith(
                            ("/", "~/", "\\\\")
                        ) or (len(candidate) > 2 and candidate[1:3] in {":\\", ":/"}):
                            return True
                    if has_path_context(nested):
                        return True
            elif isinstance(item, (list, tuple)):
                return any(has_path_context(nested) for nested in item)
            elif isinstance(item, str):
                candidate = item.strip()
                if _STRUCTURED_PATH_FIELD_RE.search(candidate):
                    return True
                if _PYTHON_STACK_RE.search(candidate) or _JS_STACK_RE.search(candidate):
                    return True
                if candidate.lower().startswith("file://") or candidate.startswith(
                    ("~/", "\\\\")
                ) or (len(candidate) > 2 and candidate[1:3] in {":\\", ":/"}):
                    return True
                if candidate.startswith("/"):
                    parts = [part for part in candidate.split("/") if part]
                    return bool(parts and parts[0].lower() not in _SAFE_ROUTE_ROOTS)
            return False

        if structured is not None and has_path_context(structured):
            return True
    if "file://" in value.lower():
        return True
    searchable_lines: list[str] = []
    for line in value.splitlines() or [value]:
        without_urls = _HTTP_URL_RE.sub("", line)
        if _PYTHON_STACK_RE.search(without_urls) or _JS_STACK_RE.search(without_urls):
            return True
        searchable_lines.append(
            _HTTP_REQUEST_TARGET_RE.sub(r"\1", without_urls, count=1)
        )
    searchable = "\n".join(searchable_lines)
    for match in _LOCAL_PATH_RE.finditer(searchable):
        candidate = match.group(0)
        if candidate.startswith(("~/", "\\\\")):
            return True
        if len(candidate) >= 3 and candidate[1:3] in {":\\", ":/"}:
            return True
        if candidate.startswith("/"):
            parts = [part for part in candidate.split("/") if part]
            prefix = searchable[max(0, match.start() - 32) : match.start()].lower()
            command_match = re.search(r"\b([a-z_]+)\s+$", prefix)
            explicit_context = bool(
                command_match
                and command_match.group(1) in _LOCAL_FILE_COMMANDS
                or re.search(
                    r"\b(?:cwd|path|file_path|file|directory)\s*[:=]\s*$", prefix
                )
            )
            if explicit_context or not parts or parts[0].lower() not in _SAFE_ROUTE_ROOTS:
                return True
    return False


def redact_local_paths(value: str) -> str:
    """Replace local path tokens while preserving useful surrounding text."""
    value = re.sub(
        r"file://[^\s\"'<>]+",
        "[local path withheld]",
        value,
        flags=re.IGNORECASE,
    )
    return _LOCAL_PATH_RE.sub(
        lambda match: (
            "[local path withheld]"
            if contains_local_path(match.group(0))
            else match.group(0)
        ),
        value,
    )


@dataclass(frozen=True)
class TrajectoryRecord:
    """One ledger row: a message event, a nested tool record, or a marker.

    Attributes:
        seq: 1-based position of this record in the rendered snapshot
            (ledger order), assigned by the projection. Not the sidecar
            ``seq`` -- legacy and compaction records have no sidecar row,
            so the render position is the only total ordering.
        kind: One of ``user`` | ``assistant`` | ``tool_call`` |
            ``tool_result`` | ``compaction``.
        turn_id: Sidecar ``turn_id`` when known; otherwise the opening
            user message's id (legacy adjacency / in-memory restore).
        message_id: Persisted message id. ``None`` for compaction markers
            and only those.
        content_preview: First 120 characters of the content, collapsed
            to a single line.
        usage: Normalized token usage for this message, or ``None``.
        step_started_at: Unix-seconds step start, or ``None`` when
            unknown. Surfaced as stored, never derived.
        first_token_at: Unix-seconds first-token time, or ``None``.
        completed_at: Unix-seconds completion time, or ``None``.
        model: Model identifier for the generating step, or ``None``.
        provider: Provider identifier for the generating step, or ``None``.
        payload: Parsed ``payload_json`` for tool records (name/args/
            result and optional ``truncated`` flag); ``None`` otherwise.
        variants: Contents of superseded variants for this record --
            active-path rendering keeps fork history visible without
            adding ledger rows. Empty tuple when none.
        depth: 0 for top-level records, 1 for tool records nested under
            their owning assistant step.
        event_id: Stable source-owner identity, independent of display order.
        source_seq: Immutable position within the source owner, when known.
        label: Human sentence-case event label.
        field_states: Per-field observed/redacted/missing-state metadata.
        sensitivity: Structured export-preflight classification.
    """

    seq: int
    kind: str
    turn_id: str
    message_id: str | None
    content_preview: str
    usage: ProviderUsage | None
    step_started_at: float | None
    first_token_at: float | None
    completed_at: float | None
    model: str | None
    provider: str | None
    payload: dict | None
    variants: tuple[str, ...]
    depth: int
    event_id: str = ""
    conversation_id: str | None = None
    source_seq: int | None = None
    label: str = ""
    status: str | None = None
    actor_kind: str | None = None
    actor_id: str | None = None
    run_id: str | None = None
    parent_event_id: str | None = None
    source_event_id: str | None = None
    replacement_event_id: str | None = None
    observed_at: float | None = None
    field_states: dict[str, str] = field(default_factory=dict)
    sensitivity: str | None = None


@dataclass(frozen=True)
class TrajectoryTurn:
    """One turn: the records from a user record to the next user record."""

    turn_id: str
    records: tuple[TrajectoryRecord, ...]


@dataclass(frozen=True)
class TrajectorySnapshot:
    """The full trajectory of a conversation, grouped into turns."""

    turns: tuple[TrajectoryTurn, ...]


# ---------------------------------------------------------------------------
# Input normalization helpers
# ---------------------------------------------------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping (dict/``sqlite3.Row``) or an object.

    The projection accepts both row shapes and model shapes; this is the
    single seam that tolerates them all.
    """
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    try:
        return obj[name]
    except Exception:  # noqa: BLE001 - sqlite3.Row/dataclass/str all differ here
        pass
    return getattr(obj, name, default)


def _parse_timestamp(value: Any) -> float | None:
    """Coerce a stored timestamp to unix seconds; ``None`` when unusable.

    Accepts numeric epochs, epoch strings, and ISO-8601 strings (naive
    values are read as UTC, matching SQLite's UTC ``CURRENT_TIMESTAMP``).
    This is representation conversion only -- it never invents a value.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def _parse_payload(raw: Any) -> dict | None:
    """Parse a sidecar ``payload_json`` string; ``None`` when absent/bad."""
    if not raw:
        return None
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _preview(content: Any) -> str:
    """Collapse content to a single line and cap at ``PREVIEW_MAX_CHARS``."""
    text = " ".join(str(content or "").split())
    return text[:PREVIEW_MAX_CHARS]


@dataclass
class _Msg:
    """Normalized view of one persisted message."""

    mid: str
    role: str
    content: str
    parent: str | None
    deleted: bool
    ts: float | None
    turn_hint: str | None
    conversation_id: str | None
    index: int


def _normalize_message(raw: Any, index: int) -> _Msg | None:
    """Build a ``_Msg`` from a row or model; ``None`` when unusable."""
    mid = _field(raw, "persisted_message_id") or _field(raw, "id")
    if not mid:
        return None
    role_raw = _field(raw, "sender") or _field(raw, "role")
    role_value = getattr(role_raw, "value", role_raw)
    role = str(role_value).lower() if role_value is not None else ""
    return _Msg(
        mid=str(mid),
        role=role,
        content=str(_field(raw, "content") or ""),
        parent=_field(raw, "parent_message_id") or None,
        deleted=bool(_field(raw, "deleted") or False),
        ts=_parse_timestamp(_field(raw, "timestamp")),
        turn_hint=_field(raw, "turn_id") or None,
        conversation_id=(
            str(_field(raw, "conversation_id"))
            if _field(raw, "conversation_id")
            else None
        ),
        index=index,
    )


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def derive_trajectory(
    messages: Iterable[Any],
    usage_by_id: Mapping[str, ProviderUsage | None],
    traj_rows: Iterable[Any],
    variant_sets: Iterable[Any],
    compaction_records: Iterable[Any],
    active_leaf_message_id: str | None = None,
    agent_runs: Iterable[Any] = (),
    agent_steps: Iterable[Any] = (),
    retrieval_runs: Iterable[Any] = (),
    diagnostic_events: Iterable[Any] = (),
) -> TrajectorySnapshot:
    """Project a conversation's persisted facts into a trajectory snapshot.

    The ledger renders the ACTIVE PATH: walking ``parent_message_id`` from
    ``active_leaf_message_id`` (the local-only ``conversations`` column) to
    the root. Soft-deleted messages (``deleted=1``) are skipped for
    rendering but still traversed, so deleting one mid-chain message never
    hides its ancestors. Tree siblings off the active chain are surfaced on
    the owning record's ``variants`` tuple instead of as rows.

    Turn boundaries are derived, never stored as rows: a turn starts at
    each ``user`` record. Compaction records render between turns with
    ``message_id=None``. Messages without sidecar rows (conversations
    predating schema v38) fall back to in-memory ``turn_id`` hints, then to
    timestamp adjacency: a message in the same calendar second as the
    preceding user message joins that user's turn (second-granularity
    legacy timestamps tie-break user-first, then by sidecar ``seq``).

    Args:
        messages: Message rows or ``ConsoleChatMessage`` models (see the
            module docstring's purity contract for the accepted shapes).
        usage_by_id: Message id -> parsed ``ProviderUsage`` (``None``
            values allowed and simply yield ``usage=None``).
        traj_rows: ``TrajectoryRowRead``-shaped sidecar rows. Rows whose
            message is not rendered (soft-deleted, off the active path, or
            absent) are ignored -- including ``tool_call``/``tool_result``
            rows keyed on such a message.
        variant_sets: ``ConsoleVariantSet``-shaped objects; superseded
            variant contents (every entry except ``selected_index``)
            attach to the assistant records of the matching turn.
        compaction_records: ``list_auxiliary_attempts``-shaped mappings
            with ``purpose == "conversation_compaction"``. Markers are
            ordered by ``started_at`` and placed after the last turn whose
            message timestamps precede them (markers that predate every
            turn lead the ledger; unplaceable ones trail it). With no
            turns at all there is no "between" -- markers are dropped.
        active_leaf_message_id: The conversation's active leaf (input
            argument; the projection never queries the DB). ``None`` or a
            dangling id degrades to rendering every non-deleted message in
            order, with no tree-derived variant suppression.
        agent_runs: Optional durable agent-run metadata rows.
        agent_steps: Optional append-only agent-step rows.
        retrieval_runs: Optional safe retrieval-provenance run summaries.

    Returns:
        The snapshot; ``turns`` is empty for an empty conversation.
    """
    normalized = []
    for index, raw in enumerate(messages):
        m = _normalize_message(raw, index)
        if m is not None:
            normalized.append(m)

    by_id = {m.mid: m for m in normalized}
    children_by_parent: dict[str | None, list[_Msg]] = {}
    for m in normalized:
        children_by_parent.setdefault(m.parent, []).append(m)

    active_ids = _active_path_ids(by_id, active_leaf_message_id)

    # Sidecar rows: message rows by message id, all other observable rows
    # by owner. Unknown kinds stay generic instead of silently displacing
    # the owning message row.
    message_rows: dict[str, Any] = {}
    sidecar_rows: dict[str, list[Any]] = {}
    replacement_owned_rows: dict[str, list[Any]] = {}
    for row in traj_rows:
        mid = _field(row, "message_id")
        kind = str(_field(row, "event_kind") or "")
        if not mid:
            continue
        if kind in _SIDECAR_ONLY_KINDS:
            continue
        if kind in _RENDERED_ROLES:
            message_rows[str(mid)] = row
        else:
            sidecar_rows.setdefault(str(mid), []).append(row)
            metadata = _parse_payload(_field(row, "payload_json")) or {}
            replacement = _optional_text(
                _field(row, "replacement_event_id")
            ) or _optional_text(metadata.get("replacement_event_id"))
            if (
                active_ids is not None
                and str(mid) not in active_ids
                and replacement is not None
                and replacement.startswith("message:")
                and replacement.removeprefix("message:") in active_ids
            ):
                replacement_owned_rows.setdefault(
                    replacement.removeprefix("message:"), []
                ).append(row)

    rendered = [
        m
        for m in normalized
        if not m.deleted
        and m.role in _RENDERED_ROLES
        and (active_ids is None or m.mid in active_ids)
    ]
    rendered.sort(key=lambda m: _sort_key(m, message_rows))

    # Events: (turn_id, record-builder) in ledger order, tool rows nested
    # directly under their owning assistant record.
    events: list[tuple[str, TrajectoryRecord]] = []
    turn_msg_times: dict[str, float] = {}
    open_turn: str | None = None
    for m in rendered:
        row = message_rows.get(m.mid)
        if row is not None:
            turn_id = str(_field(row, "turn_id") or m.mid)
        elif m.turn_hint:
            turn_id = str(m.turn_hint)
        elif m.role == KIND_USER:
            turn_id = m.mid
        elif open_turn is not None:
            turn_id = open_turn
        else:
            # Assistant-first conversation: the assistant opens its own
            # turn (mirrors the store's turn-id fallback).
            turn_id = m.mid
        if m.role == KIND_USER:
            open_turn = turn_id
        if m.ts is not None:
            prior = turn_msg_times.get(turn_id)
            if prior is None or m.ts > prior:
                turn_msg_times[turn_id] = m.ts

        row_kind = str(_field(row, "event_kind") or "") if row is not None else ""
        kind = row_kind if row_kind in _RENDERED_ROLES else m.role
        events.append(
            (
                turn_id,
                _message_record(
                    m=m,
                    kind=kind,
                    turn_id=turn_id,
                    row=row,
                    usage=usage_by_id.get(m.mid),
                    variants=_tree_variant_contents(
                        m=m,
                        children_by_parent=children_by_parent,
                        active_ids=active_ids,
                    ),
                ),
            )
        )
        if kind == KIND_ASSISTANT and row is not None:
            events.extend(
                (turn_id, record)
                for record in _model_timing_records(m=m, row=row, turn_id=turn_id)
            )
        for sidecar_row in sorted(
            sidecar_rows.get(m.mid, ()),
            key=lambda r: _as_int(_field(r, "seq")),
        ):
            events.append(
                (
                    turn_id,
                    _record_from_sidecar_event(
                        sidecar_row,
                        turn_id=turn_id,
                        owner=m,
                    ),
                )
            )
        for sidecar_row in sorted(
            replacement_owned_rows.get(m.mid, ()),
            key=lambda r: _as_int(_field(r, "seq")),
        ):
            retained = _record_from_sidecar_event(
                sidecar_row,
                turn_id=turn_id,
                owner=m,
            )
            # The row belongs durably to an off-path source, so its source
            # sequence cannot order it after the on-path replacement while
            # replacement causality orders it before that same message.
            events.append((turn_id, replace(retained, source_seq=None)))

    turns = _group_turns(events)
    turns = _insert_compaction_markers(turns, compaction_records, turn_msg_times)
    turns = _apply_variant_sets(turns, variant_sets)
    extra_records = [
        *_records_from_agent_runs(agent_runs),
        *_records_from_agent_steps(agent_steps),
        *_records_from_retrieval_runs(retrieval_runs),
        *(_record_from_sidecar_event(event) for event in diagnostic_events),
    ]
    records = [record for turn in turns for record in turn.records]
    turns = _coherent_turns(_causal_order([*records, *extra_records]))
    turns = _assign_ledger_seq(turns)
    return TrajectorySnapshot(turns=tuple(turns))


def _assign_ledger_seq(turns: list[TrajectoryTurn]) -> list[TrajectoryTurn]:
    """Stamp every record with its 1-based ledger position (render order)."""
    stamped: list[TrajectoryTurn] = []
    position = 0
    for turn in turns:
        records = []
        for record in turn.records:
            position += 1
            records.append(_with_seq(record, position))
        stamped.append(TrajectoryTurn(turn.turn_id, tuple(records)))
    return stamped


def _with_seq(record: TrajectoryRecord, seq: int) -> TrajectoryRecord:
    """Return a copy of ``record`` carrying the given ledger ``seq``."""
    return replace(record, seq=seq)


def _as_int(value: Any) -> int:
    """Best-effort int for seq sorting; unparseable sorts last."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 2**31 - 1


def _active_path_ids(
    by_id: Mapping[str, _Msg], leaf: str | None
) -> frozenset[str] | None:
    """Ids on the active path (leaf -> root), or ``None`` when unknown.

    Traversal walks the FULL map (soft-deleted nodes included) so a
    deleted mid-chain node does not hide its ancestors; a visited-set
    guards against malformed cyclic chains (mirrors the store's walk).
    A walk that ends at a non-root gap -- an ancestor id missing from
    the input map (the DB seam filters ``deleted = 0``) -- yields
    ``None`` too: the true path is unknowable, so the caller degrades
    to render-all instead of silently rendering only post-gap messages.
    """
    if not leaf or leaf not in by_id:
        return None
    chain: list[str] = []
    seen: set[str] = set()
    current: str | None = leaf
    while current is not None and current not in seen:
        if current not in by_id:
            return None
        seen.add(current)
        chain.append(current)
        current = by_id[current].parent
    return frozenset(chain)


def _sort_key(m: _Msg, message_rows: Mapping[str, Any]) -> tuple:
    """Master ledger order: timestamp, then adjacency/seq tie-breaks.

    - Timestamp primary (message rows always carry one; models without
      one keep input order among themselves, after timestamped rows).
    - Exact-timestamp ties break user-first, but ONLY for messages with
      no turn identity of their own (no sidecar row, no ``turn_id``
      hint): second-granularity legacy rows persist a fast reply with the
      same stamp as its user message, and the user message opens the
      turn, so it must sort first. Messages that already know their turn
      never reorder.
    - Sidecar ``seq`` breaks remaining ties (authoritative monotonic
      order), then input position.
    """
    has_identity = message_rows.get(m.mid) is not None or m.turn_hint is not None
    user_first = 0 if (m.role == KIND_USER or has_identity) else 1
    row = message_rows.get(m.mid)
    seq = _as_int(_field(row, "seq")) if row is not None else _NO_SEQ
    return (
        m.ts is None,  # timestamped rows first
        m.ts if m.ts is not None else 0.0,
        user_first,
        seq,
        m.index,
    )


def _tree_variant_contents(
    m: _Msg,
    children_by_parent: Mapping[str | None, list[_Msg]],
    active_ids: frozenset[str] | None,
) -> tuple[str, ...]:
    """Contents of this node's tree siblings that are off the active path.

    A sibling sharing ``m``'s parent but not on the active chain is a
    superseded fork at the same tree position; its content surfaces on
    ``m``'s record instead of as a ledger row. Only direct siblings
    count -- deeper divergence inside a sibling's subtree is its history,
    not a variant of ``m``. Requires a known active path; with none
    (``active_leaf_message_id`` unset/dangling) there is no chain to be
    off of, so no tree variants are derived.
    """
    if active_ids is None or m.parent is None:
        return ()
    siblings = [
        s
        for s in children_by_parent.get(m.parent, ())
        if s.mid != m.mid
        and not s.deleted
        and s.role in _RENDERED_ROLES
        and s.mid not in active_ids
    ]
    siblings.sort(key=lambda s: (s.ts is None, s.ts or 0.0, s.index))
    return tuple(s.content for s in siblings)


def _message_record(
    *,
    m: _Msg,
    kind: str,
    turn_id: str,
    row: Any,
    usage: ProviderUsage | None,
    variants: tuple[str, ...],
) -> TrajectoryRecord:
    """Build a top-level user/assistant record from a message + its row."""
    conversation_id = (
        _optional_text(_field(row, "conversation_id")) or m.conversation_id
    )
    system_message = kind == KIND_SYSTEM
    metadata = _parse_payload(_field(row, "payload_json")) or {}
    completion_observed_at = _parse_timestamp(_field(row, "completed_at"))
    completed_v2_assistant = (
        kind == KIND_ASSISTANT
        and metadata.get("trace_version") == 2
        and metadata.get("model_status") == "completed"
        and completion_observed_at is not None
    )
    return TrajectoryRecord(
        seq=0,  # assigned by the final pass
        kind=kind,
        turn_id=turn_id,
        message_id=m.mid,
        content_preview=(
            "System context attached" if system_message else _preview(m.content)
        ),
        usage=usage,
        step_started_at=_field(row, "step_started_at") if row is not None else None,
        first_token_at=_field(row, "first_token_at") if row is not None else None,
        completed_at=_field(row, "completed_at") if row is not None else None,
        model=_field(row, "model") if row is not None else None,
        provider=_field(row, "provider") if row is not None else None,
        payload=None,
        variants=variants,
        depth=0,
        event_id=f"message:{m.mid}",
        conversation_id=conversation_id,
        source_seq=_optional_int(_field(row, "seq")),
        label=_event_label(kind),
        status=_optional_text(_field(row, "status")) or "complete",
        actor_kind=kind,
        actor_id=kind,
        parent_event_id=(
            f"model-timing:{m.mid}:completed"
            if completed_v2_assistant
            else f"message:{m.parent}"
            if m.parent
            else None
        ),
        source_event_id=_optional_text(_field(row, "source_event_id")),
        replacement_event_id=_optional_text(_field(row, "replacement_event_id")),
        observed_at=completion_observed_at if completed_v2_assistant else m.ts,
        field_states=_field_state_map(
            row,
            default={"content_preview": "omitted" if system_message else "observed"},
        ),
        sensitivity=(
            _optional_text(_field(row, "sensitivity"))
            or ("system_context" if system_message else "conversation_content")
        ),
    )


def _model_timing_records(
    *, m: _Msg, row: Any, turn_id: str
) -> tuple[TrajectoryRecord, ...]:
    """Expose the three model observations already recorded by one message row."""
    metadata = _parse_payload(_field(row, "payload_json")) or {}
    if metadata.get("trace_version") != 2:
        return ()
    conversation_id = (
        _optional_text(_field(row, "conversation_id")) or m.conversation_id
    )
    source_seq = _optional_int(_field(row, "seq"))
    observations = [
        (
            "model_request_started",
            "started",
            _parse_timestamp(_field(row, "step_started_at")),
            f"message:{m.parent}" if m.parent else None,
        ),
        (
            "model_first_token",
            "streaming",
            _parse_timestamp(_field(row, "first_token_at")),
            f"model-timing:{m.mid}:started",
        ),
        (
            "model_response_completed",
            "completed",
            _parse_timestamp(_field(row, "completed_at")),
            f"model-timing:{m.mid}:first-token",
        ),
    ]
    if metadata.get("model_status") != "completed":
        observations.pop()
    suffixes = ("started", "first-token", "completed")[: len(observations)]
    records: list[TrajectoryRecord] = []
    for suffix, (kind, status, observed_at, parent_event_id) in zip(
        suffixes, observations
    ):
        if observed_at is None:
            continue
        base = TrajectoryRecord(
            seq=0,
            kind=kind,
            turn_id=turn_id,
            message_id=m.mid,
            content_preview=kind.replace("_", " "),
            usage=None,
            step_started_at=observed_at,
            first_token_at=(observed_at if kind == "model_first_token" else None),
            completed_at=(observed_at if kind == "model_response_completed" else None),
            model=_optional_text(_field(row, "model")),
            provider=_optional_text(_field(row, "provider")),
            payload=None,
            variants=(),
            depth=1,
            event_id=f"model-timing:{m.mid}:{suffix}",
            conversation_id=conversation_id,
            source_seq=source_seq,
            label=_event_label(kind),
            status=status,
            actor_kind="model",
            actor_id=_optional_text(_field(row, "provider")) or "model",
            parent_event_id=parent_event_id,
            observed_at=observed_at,
            field_states={"observed_at": "observed", "payload": "omitted"},
            sensitivity="diagnostic",
        )
        records.append(base)
    return tuple(records)


def _record_from_sidecar_event(
    row: Any,
    *,
    turn_id: str | None = None,
    owner: _Msg | None = None,
) -> TrajectoryRecord:
    """Normalize one known or future trajectory-sidecar event."""
    payload = _parse_payload(_field(row, "payload_json"))
    metadata = payload or {}
    kind = str(_field(row, "event_kind") or "event")
    message_id = _optional_text(_field(row, "message_id"))
    conversation_id = _optional_text(_field(row, "conversation_id"))
    source_seq = _optional_int(_field(row, "seq"))
    actual_turn_id = (
        _optional_text(_field(row, "turn_id"))
        or turn_id
        or message_id
        or conversation_id
        or "trace"
    )
    if kind == KIND_USER_FEEDBACK:
        preview = _feedback_preview(payload)
    elif kind in _TOOL_KINDS:
        preview = _tool_preview(payload)
    else:
        preview = _preview(
            _field(row, "summary")
            or (payload.get("summary") if payload else "")
            or kind.replace("_", " ")
        )
    return TrajectoryRecord(
        seq=0,  # assigned by the final pass
        kind=kind,
        turn_id=actual_turn_id,
        message_id=message_id,
        content_preview=preview,
        usage=None,
        step_started_at=_field(row, "step_started_at"),
        first_token_at=_field(row, "first_token_at"),
        completed_at=_field(row, "completed_at"),
        model=_field(row, "model"),
        provider=_field(row, "provider"),
        payload=payload,
        variants=(),
        depth=1 if message_id else 0,
        event_id=_sidecar_event_id(row),
        conversation_id=conversation_id,
        source_seq=source_seq,
        label=_event_label(kind),
        status=(
            _optional_text(_field(row, "status"))
            or _optional_text(metadata.get("status"))
            or "observed"
        ),
        actor_kind=_optional_text(_field(row, "actor_kind")),
        actor_id=_optional_text(_field(row, "actor_id")),
        run_id=_optional_text(_field(row, "run_id")),
        parent_event_id=(
            _optional_text(_field(row, "parent_event_id"))
            or _optional_text(metadata.get("parent_event_id"))
            or (f"message:{message_id}" if message_id else None)
        ),
        source_event_id=(
            _optional_text(_field(row, "source_event_id"))
            or _optional_text(metadata.get("source_event_id"))
        ),
        replacement_event_id=(
            _optional_text(_field(row, "replacement_event_id"))
            or _optional_text(metadata.get("replacement_event_id"))
        ),
        observed_at=_first_timestamp(
            _field(row, "step_started_at"),
            _field(row, "first_token_at"),
            _field(row, "completed_at"),
            owner.ts if owner is not None else None,
        ),
        field_states=_field_state_map(
            row,
            default=(
                {
                    str(key): str(value)
                    for key, value in metadata["field_states"].items()
                }
                if isinstance(metadata.get("field_states"), Mapping)
                else {"payload": "observed" if payload is not None else "not_available"}
            ),
        ),
        sensitivity=(
            _optional_text(_field(row, "sensitivity"))
            or _optional_text(metadata.get("sensitivity"))
            or ("tool_content" if kind in _TOOL_KINDS else None)
            or ("conversation_content" if payload is not None else "diagnostic")
        ),
    )


_FEEDBACK_ACTION_LABELS = {
    "request-changes": "Request changes",
    "lgm": "LGTM",
    "comment": "Comment",
}


def _feedback_preview(payload: dict | None) -> str:
    """Single-line preview of a selection-feedback payload.

    Shows what the user SAID where they said anything -- the comment is the
    reviewer's own words and the reason the record exists. With no comment
    (LGTM, or Request-changes submitted bare) the quote is the only content
    there is, so it stands in.
    """
    if not payload:
        return ""
    action = str(payload.get("action") or "")
    label = _FEEDBACK_ACTION_LABELS.get(action, action or "Feedback")
    detail = str(payload.get("comment") or payload.get("quote") or "").strip()
    detail = " ".join(detail.split())
    return f"{label}: {detail}" if detail else label


def _tool_preview(payload: dict | None) -> str:
    """Single-line preview of a tool payload: ``name -> result``."""
    if not payload:
        return ""
    name = str(payload.get("name") or "").strip()
    result = payload.get("result")
    body = _preview(result)
    if name and body:
        return f"{name} -> {body}"[:PREVIEW_MAX_CHARS]
    return body or name[:PREVIEW_MAX_CHARS]


_EVENT_LABELS = {
    KIND_USER: "User message",
    KIND_SYSTEM: "System context",
    KIND_ASSISTANT: "Assistant message",
    KIND_TOOL_CALL: "Tool call",
    KIND_TOOL_RESULT: "Tool result",
    KIND_COMPACTION: "Compaction",
    KIND_USER_FEEDBACK: "User feedback",
    "model": "Model response",
    "spawn": "Agent spawn",
    "agent_run": "Agent run",
    "retrieval_run": "Retrieval run",
}


def _event_label(kind: str) -> str:
    """Human sentence-case label for one storage event kind."""
    return _EVENT_LABELS.get(kind, kind.replace("_", " ").strip().capitalize())


def _optional_text(value: Any) -> str | None:
    """Return a non-empty string, or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    """Return an integer owner sequence, or ``None`` when unavailable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_timestamp(*values: Any) -> float | None:
    """Return the first observed timestamp in the supplied precedence."""
    for value in values:
        parsed = _parse_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def _field_state_map(
    source: Any,
    *,
    default: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy structured per-field states from a plain source object."""
    states = _field(source, "field_states")
    if isinstance(states, Mapping):
        return {str(key): str(value) for key, value in states.items()}
    return dict(default or {})


def _sidecar_event_id(row: Any) -> str:
    """Stable identifier for a trajectory-sidecar owner row."""
    explicit = _optional_text(_field(row, "event_id"))
    if explicit is None:
        payload = _parse_payload(_field(row, "payload_json"))
        explicit = _optional_text(payload.get("event_id")) if payload else None
    if explicit:
        return explicit
    conversation_id = _optional_text(_field(row, "conversation_id"))
    message_id = _optional_text(_field(row, "message_id")) or "unknown"
    source_seq = _optional_int(_field(row, "seq"))
    owner = conversation_id or message_id
    suffix = (
        str(source_seq)
        if source_seq is not None
        else _stable_digest(
            message_id,
            conversation_id,
            _field(row, "turn_id"),
            _field(row, "event_kind"),
            _field(row, "step_started_at"),
            _field(row, "first_token_at"),
            _field(row, "completed_at"),
            _field(row, "payload_json"),
        )
    )
    return f"trajectory:{owner}:{suffix}"


def _stable_digest(*values: Any) -> str:
    """Return a compact deterministic identity for immutable source fields."""
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _records_from_agent_runs(runs: Iterable[Any]) -> list[TrajectoryRecord]:
    """Adapt durable AgentRunsDB run metadata into Trace records."""
    records: list[TrajectoryRecord] = []
    for run in runs:
        run_id = _optional_text(_field(run, "id") or _field(run, "run_id"))
        if not run_id:
            continue
        conversation_id = _optional_text(_field(run, "conversation_id"))
        actor_kind = _optional_text(_field(run, "agent_kind")) or "agent"
        parent_run_id = _optional_text(_field(run, "parent_run_id"))
        task = _optional_text(_field(run, "task")) or ""
        created_at = _parse_timestamp(_field(run, "created_at"))
        status = _optional_text(_field(run, "status")) or "unknown"
        run_states = _field_state_map(
            run,
            default={
                "result": (
                    "observed" if _field(run, "result") is not None else "not_available"
                )
            },
        )
        if task:
            run_states["task"] = "omitted"
        records.append(
            TrajectoryRecord(
                seq=0,
                kind="agent_run",
                turn_id=(
                    _optional_text(_field(run, "turn_id"))
                    or conversation_id
                    or f"run:{run_id}"
                ),
                message_id=_optional_text(_field(run, "assistant_message_id")),
                content_preview=_preview(task or f"{actor_kind} run"),
                usage=None,
                step_started_at=created_at,
                first_token_at=None,
                completed_at=(
                    _parse_timestamp(_field(run, "updated_at"))
                    if status in {"done", "error", "stuck", "cancelled", "superseded"}
                    else None
                ),
                model=_optional_text(_field(run, "model")),
                provider=_optional_text(_field(run, "provider")),
                payload=None,
                variants=(),
                depth=0,
                event_id=f"agent-run:{run_id}",
                conversation_id=conversation_id,
                source_seq=_optional_int(_field(run, "source_seq")),
                label="Agent run",
                status=status,
                actor_kind=actor_kind,
                actor_id=(
                    _optional_text(_field(run, "agent_definition"))
                    or _optional_text(_field(run, "agent_definition_id"))
                    or run_id
                ),
                run_id=run_id,
                parent_event_id=(
                    _optional_text(_field(run, "parent_event_id"))
                    or _optional_text(_field(run, "spawn_event_id"))
                    or (f"agent-run:{parent_run_id}" if parent_run_id else None)
                ),
                source_event_id=(
                    _optional_text(_field(run, "source_event_id"))
                    or (
                        f"agent-run:{_field(run, 'resumed_from_run_id')}"
                        if _field(run, "resumed_from_run_id")
                        else None
                    )
                ),
                replacement_event_id=_optional_text(
                    _field(run, "replacement_event_id")
                ),
                observed_at=created_at,
                field_states=run_states,
                sensitivity=(
                    "conversation_content"
                    if task
                    else (_optional_text(_field(run, "sensitivity")) or "diagnostic")
                ),
            )
        )
    return records


def _records_from_agent_steps(steps: Iterable[Any]) -> list[TrajectoryRecord]:
    """Adapt append-only AgentRunsDB step rows into Trace records."""
    records: list[TrajectoryRecord] = []
    for step in steps:
        run_id = _optional_text(_field(step, "run_id"))
        step_index = _optional_int(_field(step, "index"))
        if step_index is None:
            step_index = _optional_int(_field(step, "seq"))
        if not run_id or step_index is None:
            continue
        owner_seq = _optional_int(_field(step, "owner_seq"))
        source_seq = owner_seq if owner_seq is not None else step_index
        kind = _optional_text(_field(step, "kind")) or "agent_step"
        summary = _optional_text(_field(step, "summary")) or ""
        created_at = _parse_timestamp(_field(step, "created_at"))
        source_payload = _field(step, "payload")
        source_payload = source_payload if isinstance(source_payload, Mapping) else {}
        tool_name = _field(step, "tool_name") or source_payload.get("tool_name")
        outcome = _optional_text(_field(step, "tool_outcome"))
        outcome = outcome or _optional_text(source_payload.get("tool_outcome"))
        payload = {
            key: value
            for key, value in (("tool_name", tool_name), ("tool_outcome", outcome))
            if value not in (None, "")
        }
        has_tool_content = any(
            (_field(step, key) or source_payload.get(key)) not in (None, "")
            for key in ("args", "result")
        )
        default_states = {"summary": "observed" if summary else "not_available"}
        for key in ("args", "result", "tool_outcome"):
            value = _field(step, key) or source_payload.get(key)
            default_states[key] = "not_available"
            if value not in (None, ""):
                default_states[key] = (
                    "omitted" if key in {"args", "result"} else "observed"
                )
        records.append(
            TrajectoryRecord(
                seq=0,
                kind=kind,
                turn_id=(
                    _optional_text(_field(step, "turn_id"))
                    or _optional_text(_field(step, "conversation_id"))
                    or f"run:{run_id}"
                ),
                message_id=_optional_text(_field(step, "message_id")),
                content_preview=_preview(summary or kind.replace("_", " ")),
                usage=None,
                step_started_at=created_at,
                first_token_at=None,
                completed_at=created_at,
                model=_optional_text(_field(step, "model")),
                provider=_optional_text(_field(step, "provider")),
                payload=dict(payload) if payload else None,
                variants=(),
                depth=1,
                event_id=f"agent-step:{run_id}:{step_index}",
                conversation_id=_optional_text(_field(step, "conversation_id")),
                source_seq=source_seq,
                label=_event_label(kind),
                status=_optional_text(_field(step, "status")) or outcome or "observed",
                actor_kind=_optional_text(_field(step, "actor_kind")) or "agent",
                actor_id=_optional_text(_field(step, "actor_id")) or run_id,
                run_id=run_id,
                parent_event_id=(
                    _optional_text(_field(step, "parent_event_id"))
                    or (
                        f"agent-step:{run_id}:{_field(step, 'parent_step_index')}"
                        if _field(step, "parent_step_index") is not None
                        else None
                    )
                    or f"agent-run:{run_id}"
                ),
                source_event_id=(
                    _optional_text(_field(step, "source_event_id"))
                    or (
                        f"agent-step:{run_id}:{_field(step, 'source_step_index')}"
                        if _field(step, "source_step_index") is not None
                        else None
                    )
                ),
                replacement_event_id=_optional_text(
                    _field(step, "replacement_event_id")
                ),
                observed_at=created_at,
                field_states=_field_state_map(
                    step,
                    default=default_states,
                ),
                sensitivity=(
                    _optional_text(_field(step, "sensitivity"))
                    or ("conversation_content" if kind == "model" else None)
                    or ("tool_content" if has_tool_content else "diagnostic")
                ),
            )
        )
    return records


def _records_from_retrieval_runs(runs: Iterable[Any]) -> list[TrajectoryRecord]:
    """Adapt safe citation/retrieval run summaries into Trace records."""
    records: list[TrajectoryRecord] = []
    for run in runs:
        run_id = _optional_text(_field(run, "run_id") or _field(run, "id"))
        if not run_id:
            continue
        started_at = _parse_timestamp(
            _field(run, "started_at") or _field(run, "created_at")
        )
        ended_at = _parse_timestamp(_field(run, "ended_at"))
        stage = _optional_text(_field(run, "stage")) or "retrieval"
        conversation_id = _optional_text(_field(run, "conversation_id"))
        base = TrajectoryRecord(
            seq=0,
            kind="retrieval_run",
            turn_id=(
                _optional_text(_field(run, "turn_id"))
                or conversation_id
                or f"retrieval:{run_id}"
            ),
            message_id=_optional_text(_field(run, "message_id")),
            content_preview=_preview(stage.replace("_", " ")),
            usage=None,
            step_started_at=started_at,
            first_token_at=None,
            completed_at=ended_at,
            model=None,
            provider=None,
            payload={"stage": stage},
            variants=(),
            depth=0,
            event_id=f"retrieval-run:{run_id}",
            conversation_id=conversation_id,
            source_seq=_optional_int(
                _field(run, "run_ordinal") or _field(run, "source_seq")
            ),
            label="Retrieval run",
            status=(
                _optional_text(_field(run, "status"))
                or ("complete" if ended_at is not None else "running")
            ),
            actor_kind="retrieval",
            actor_id=_optional_text(_field(run, "actor_id")) or "retrieval",
            run_id=run_id,
            parent_event_id=_optional_text(_field(run, "parent_event_id")),
            source_event_id=_optional_text(_field(run, "source_event_id")),
            replacement_event_id=_optional_text(_field(run, "replacement_event_id")),
            observed_at=started_at,
            field_states=_field_state_map(
                run,
                default={"payload": "omitted"},
            ),
            sensitivity=_optional_text(_field(run, "sensitivity"))
            or "retrieval_metadata",
        )
        records.append(base)
    return records


def _causal_order(records: Iterable[TrajectoryRecord]) -> list[TrajectoryRecord]:
    """Deterministically order records while keeping causes before effects."""
    unique_records = _unique_event_ids(records)
    by_id = {record.event_id: record for record in unique_records}
    if not by_id:
        return []

    edges: dict[str, set[str]] = {event_id: set() for event_id in by_id}
    indegree = {event_id: 0 for event_id in by_id}

    def add_edge(before: str | None, after: str | None) -> None:
        if (
            not before
            or not after
            or before == after
            or before not in by_id
            or after not in by_id
            or after in edges[before]
        ):
            return
        edges[before].add(after)
        indegree[after] += 1

    for record in by_id.values():
        add_edge(record.parent_event_id, record.event_id)
        add_edge(record.source_event_id, record.event_id)
        add_edge(record.event_id, record.replacement_event_id)
        if record.kind == "retrieval_run" and record.message_id:
            add_edge(record.event_id, f"message:{record.message_id}")

    owner_sequences: dict[tuple[str, str], list[TrajectoryRecord]] = {}
    for record in by_id.values():
        if record.source_seq is None:
            continue
        if record.event_id.startswith("agent-step:") and record.run_id:
            owner = ("agent-step", record.run_id)
        elif (
            record.event_id.startswith(("message:", "trajectory:"))
            and record.conversation_id
        ):
            owner = ("trajectory", record.conversation_id)
        else:
            continue
        owner_sequences.setdefault(owner, []).append(record)
    for owner_records in owner_sequences.values():
        sequence_groups: dict[int, list[TrajectoryRecord]] = {}
        for record in owner_records:
            sequence_groups.setdefault(record.source_seq or 0, []).append(record)
        ordered_groups = [sequence_groups[seq] for seq in sorted(sequence_groups)]
        for before_group, after_group in zip(ordered_groups, ordered_groups[1:]):
            for before in before_group:
                for after in after_group:
                    add_edge(before.event_id, after.event_id)

    def key(event_id: str) -> tuple[Any, ...]:
        record = by_id[event_id]
        return (
            record.observed_at is None,
            record.observed_at or 0.0,
            0
            if record.event_id.startswith("message:") and record.kind == KIND_USER
            else 1,
            record.source_seq is None,
            record.source_seq or 0,
            event_id,
        )

    components = _strong_components(by_id, edges)
    component_by_id = {
        event_id: index
        for index, component in enumerate(components)
        for event_id in component
    }
    component_edges: dict[int, set[int]] = {i: set() for i in range(len(components))}
    component_indegree = {i: 0 for i in range(len(components))}
    for before, children in edges.items():
        for after in children:
            source = component_by_id[before]
            target = component_by_id[after]
            if source == target or target in component_edges[source]:
                continue
            component_edges[source].add(target)
            component_indegree[target] += 1

    def component_key(index: int) -> tuple[Any, ...]:
        return min(key(event_id) for event_id in components[index])

    ready = [
        (component_key(index), index)
        for index, degree in component_indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    ordered_ids: list[str] = []
    while ready:
        _, index = heapq.heappop(ready)
        ordered_ids.extend(sorted(components[index], key=key))
        for child in sorted(component_edges[index], key=component_key):
            component_indegree[child] -= 1
            if component_indegree[child] == 0:
                heapq.heappush(ready, (component_key(child), child))
    return [by_id[event_id] for event_id in ordered_ids]


def _strong_components(
    nodes: Mapping[str, Any], edges: Mapping[str, set[str]]
) -> list[list[str]]:
    """Return deterministic SCCs without consuming Python's call stack."""
    visited: set[str] = set()
    finish: list[str] = []
    for root in sorted(nodes):
        if root in visited:
            continue
        visited.add(root)
        stack: list[tuple[str, Any]] = [(root, iter(sorted(edges[root])))]
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
            except StopIteration:
                stack.pop()
                finish.append(node)
                continue
            if child not in visited:
                visited.add(child)
                stack.append((child, iter(sorted(edges[child]))))

    reverse_edges: dict[str, set[str]] = {node: set() for node in nodes}
    for before, children in edges.items():
        for after in children:
            reverse_edges[after].add(before)
    components: list[list[str]] = []
    assigned: set[str] = set()
    for root in reversed(finish):
        if root in assigned:
            continue
        component: list[str] = []
        stack = [(root, False)]
        assigned.add(root)
        while stack:
            node, _ = stack.pop()
            component.append(node)
            for parent in reversed(sorted(reverse_edges[node])):
                if parent not in assigned:
                    assigned.add(parent)
                    stack.append((parent, False))
        components.append(component)
    return components


def _unique_event_ids(records: Iterable[TrajectoryRecord]) -> list[TrajectoryRecord]:
    """Keep colliding source rows by assigning deterministic derived identities."""
    groups: dict[str, list[TrajectoryRecord]] = {}
    for record in records:
        base = record.event_id or f"trace-event:{_record_digest(record)}"
        groups.setdefault(base, []).append(record)

    ambiguous = {base for base, group in groups.items() if len(group) > 1}
    unique: list[TrajectoryRecord] = []
    for base in sorted(groups):
        occurrences: dict[str, int] = {}
        for record in sorted(groups[base], key=lambda item: _record_digest(item)):
            digest = _record_digest(record)
            occurrences[digest] = occurrences.get(digest, 0) + 1
            event_id = base
            if base in ambiguous:
                event_id = f"{base}:collision:{digest}:{occurrences[digest]}"
                states = dict(record.field_states)
                states["event_id"] = "capture_failed"
                record = replace(record, field_states=states)
            unique.append(replace(record, event_id=event_id))
    resolved: list[TrajectoryRecord] = []
    for record in unique:
        changes: dict[str, Any] = {}
        states = dict(record.field_states)
        for field_name in (
            "parent_event_id",
            "source_event_id",
            "replacement_event_id",
        ):
            if getattr(record, field_name) in ambiguous:
                changes[field_name] = None
                states[field_name] = "capture_failed"
        if changes:
            changes["field_states"] = states
            record = replace(record, **changes)
        resolved.append(record)
    return resolved


def _record_digest(record: TrajectoryRecord) -> str:
    """Hash a record's source envelope without its mutable display position."""
    data = asdict(record)
    data.pop("seq", None)
    return _stable_digest(data)


def _coherent_turns(records: Iterable[TrajectoryRecord]) -> list[TrajectoryTurn]:
    """Group a causal stream into one causally ordered block per turn."""
    ordered = list(records)
    buckets: dict[str, list[TrajectoryRecord]] = {}
    event_turn: dict[str, str] = {}
    turn_position: dict[str, int] = {}
    for position, record in enumerate(ordered):
        turn_id = record.turn_id or record.conversation_id or "trace"
        buckets.setdefault(turn_id, []).append(record)
        event_turn[record.event_id] = turn_id
        turn_position.setdefault(turn_id, position)

    edges: dict[str, set[str]] = {turn_id: set() for turn_id in buckets}
    indegree = {turn_id: 0 for turn_id in buckets}

    def add_event_edge(before: str | None, after: str | None) -> None:
        before_turn = event_turn.get(before or "")
        after_turn = event_turn.get(after or "")
        if (
            before_turn is None
            or after_turn is None
            or before_turn == after_turn
            or after_turn in edges[before_turn]
        ):
            return
        edges[before_turn].add(after_turn)
        indegree[after_turn] += 1

    for record in ordered:
        add_event_edge(record.parent_event_id, record.event_id)
        add_event_edge(record.source_event_id, record.event_id)
        add_event_edge(record.event_id, record.replacement_event_id)
        if record.kind == "retrieval_run" and record.message_id:
            add_event_edge(record.event_id, f"message:{record.message_id}")

    owner_sequences: dict[tuple[str, str], list[TrajectoryRecord]] = {}
    for record in ordered:
        if record.source_seq is None or not record.conversation_id:
            continue
        if record.event_id.startswith(("message:", "trajectory:")):
            owner_sequences.setdefault(
                ("trajectory", record.conversation_id), []
            ).append(record)
    for owner_records in owner_sequences.values():
        sequence_groups: dict[int, list[TrajectoryRecord]] = {}
        for record in owner_records:
            sequence_groups.setdefault(record.source_seq or 0, []).append(record)
        ordered_groups = [sequence_groups[seq] for seq in sorted(sequence_groups)]
        for before_group, after_group in zip(ordered_groups, ordered_groups[1:]):
            for before in before_group:
                for after in after_group:
                    add_event_edge(before.event_id, after.event_id)

    ready = [
        (turn_position[turn_id], turn_id)
        for turn_id, degree in indegree.items()
        if degree == 0
    ]
    heapq.heapify(ready)
    turn_ids: list[str] = []
    while ready:
        _, turn_id = heapq.heappop(ready)
        turn_ids.append(turn_id)
        for child_id in sorted(edges[turn_id]):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                heapq.heappush(ready, (turn_position[child_id], child_id))
    if len(turn_ids) != len(buckets):
        # Turn contraction can introduce a cycle even when the event graph is
        # valid. Preserve the honest event order in repeated turn segments.
        return _group_turns(
            [
                (record.turn_id or record.conversation_id or "trace", record)
                for record in ordered
            ]
        )
    return [TrajectoryTurn(turn_id, tuple(buckets[turn_id])) for turn_id in turn_ids]


def _group_turns(events: list[tuple[str, TrajectoryRecord]]) -> list[TrajectoryTurn]:
    """Fold the ordered event stream into contiguous turns."""
    turns: list[TrajectoryTurn] = []
    current_id: str | None = None
    current_records: list[TrajectoryRecord] = []
    for turn_id, record in events:
        if turn_id != current_id:
            if current_id is not None:
                turns.append(TrajectoryTurn(current_id, tuple(current_records)))
            current_id = turn_id
            current_records = []
        current_records.append(record)
    if current_id is not None:
        turns.append(TrajectoryTurn(current_id, tuple(current_records)))
    return turns


def _insert_compaction_markers(
    turns: list[TrajectoryTurn],
    compaction_records: Iterable[Any],
    turn_msg_times: Mapping[str, float],
) -> list[TrajectoryTurn]:
    """Place compaction markers between turns (trailing the turn they follow).

    Placement keys off message timestamps only -- the persisted message
    time is the message's canonical DB fact, immune to the sidecar's
    nullable timing. Markers sort by ``started_at``; one predating every
    turn leads the ledger; one after the last turn (or without a usable
    time) trails it. With no turns there is no "between": markers drop.
    """
    markers = [
        rec
        for rec in compaction_records
        if str(_field(rec, "purpose") or _COMPACTION_PURPOSE) == _COMPACTION_PURPOSE
    ]
    if not markers or not turns:
        return turns

    markers = sorted(
        enumerate(markers),
        key=lambda pair: (
            _parse_timestamp(_field(pair[1], "started_at")) or 0.0,
            pair[0],
        ),
    )
    turn_ends = [turn_msg_times.get(turn.turn_id) for turn in turns]

    # buckets[i] = markers trailing turn i (in marker order).
    buckets: list[list[Any]] = [[] for _ in turns]
    lead: list[Any] = []
    for _, rec in markers:
        when = _parse_timestamp(_field(rec, "started_at"))
        home = len(turns) - 1  # untimeable / after everything
        if when is not None:
            home = -1  # before everything until proven otherwise
            for i, end in enumerate(turn_ends):
                if end is not None and end <= when:
                    home = i
        (lead if home == -1 else buckets[home]).append(rec)

    result: list[TrajectoryTurn] = []
    for i, turn in enumerate(turns):
        records: list[TrajectoryRecord] = []
        if i == 0:
            for rec in lead:
                records.extend(_marker_records(rec, turn_id=turn.turn_id))
        records.extend(turn.records)
        for rec in buckets[i]:
            records.extend(_marker_records(rec, turn_id=turn.turn_id))
        result.append(TrajectoryTurn(turn.turn_id, tuple(records)))
    return result


def _marker_records(rec: Any, *, turn_id: str) -> tuple[TrajectoryRecord, ...]:
    """Keep the v1 marker unless its owner opts into v2 lifecycle records."""
    marker = _marker_record(rec, turn_id=turn_id)
    if not bool(_field(rec, "trace_lifecycle")):
        return (marker,)
    operation_id = marker.event_id.removeprefix("compaction:")
    started = replace(
        marker,
        kind="compaction_started",
        event_id=f"compaction:{operation_id}:started",
        label="Compaction started",
        status="started",
        completed_at=None,
        parent_event_id=None,
        field_states={"payload": "not_available", "observed_at": "observed"},
    )
    status = marker.status or "unknown"
    if status not in {"succeeded", "failed", "cancelled"}:
        return (marker, started)
    terminal_kind = {
        "succeeded": "compaction_completed",
        "failed": "compaction_failed",
        "cancelled": "compaction_cancelled",
    }.get(status, "compaction_outcome")
    terminal = replace(
        marker,
        kind=terminal_kind,
        event_id=f"compaction:{operation_id}:outcome",
        label=_event_label(terminal_kind),
        parent_event_id=started.event_id,
        observed_at=marker.completed_at,
        field_states={
            "payload": "not_available",
            "observed_at": (
                "observed" if marker.completed_at is not None else "not_available"
            ),
        },
    )
    return (marker, started, terminal)


def _marker_record(rec: Any, *, turn_id: str) -> TrajectoryRecord:
    """Build a between-turn compaction marker record."""
    status = str(_field(rec, "status") or "")
    usage = ProviderUsage.from_json(_field(rec, "provider_usage_json"))
    conversation_id = _optional_text(_field(rec, "conversation_id"))
    started_at = _parse_timestamp(_field(rec, "started_at"))
    operation_id = _optional_text(_field(rec, "operation_id")) or _stable_digest(
        conversation_id,
        _field(rec, "purpose"),
        _field(rec, "started_at"),
        _field(rec, "finished_at"),
        status,
        _field(rec, "provider"),
        _field(rec, "model"),
    )
    return TrajectoryRecord(
        seq=0,  # assigned by the final pass
        kind=KIND_COMPACTION,
        turn_id=turn_id,
        message_id=None,
        content_preview=f"compaction: {status}" if status else "compaction",
        usage=usage,
        step_started_at=started_at,
        first_token_at=None,  # compaction has no token boundary
        completed_at=_parse_timestamp(_field(rec, "finished_at")),
        model=_field(rec, "model") or None,
        provider=_field(rec, "provider") or None,
        payload=None,
        variants=(),
        depth=0,
        event_id=f"compaction:{operation_id}",
        conversation_id=conversation_id,
        label="Compaction",
        status=status or "unknown",
        actor_kind="context",
        actor_id="compaction",
        observed_at=started_at,
        field_states={"payload": "not_available"},
        sensitivity="diagnostic",
    )


def _apply_variant_sets(
    turns: list[TrajectoryTurn], variant_sets: Iterable[Any]
) -> list[TrajectoryTurn]:
    """Attach superseded variant-set contents to their turn's assistant records.

    A set's superseded contents are every entry except ``selected_index``
    (the selected entry is the current rendering). The set identifies a
    turn, not a message, so contents attach to the turn's assistant
    records -- merged after tree-derived variants, deduplicated in order.
    """
    sets = list(variant_sets)
    if not sets:
        return turns
    superseded_by_turn: dict[str, tuple[str, ...]] = {}
    for vs in sets:
        turn_id = _field(vs, "turn_id")
        if not turn_id:
            continue
        selected = _as_int(_field(vs, "selected_index", 0))
        contents = tuple(
            content
            for index, item in enumerate(_field(vs, "variants") or ())
            if index != selected
            for content in (_variant_content(item),)
            if content is not None
        )
        superseded_by_turn.setdefault(str(turn_id), ())
        superseded_by_turn[str(turn_id)] = superseded_by_turn[str(turn_id)] + contents

    if not superseded_by_turn:
        return turns

    result: list[TrajectoryTurn] = []
    for turn in turns:
        extra = superseded_by_turn.get(turn.turn_id)
        if not extra:
            result.append(turn)
            continue
        records = tuple(
            _merge_variants(record, extra) if record.kind == KIND_ASSISTANT else record
            for record in turn.records
        )
        result.append(TrajectoryTurn(turn.turn_id, records))
    return result


def _variant_content(item: Any) -> str | None:
    """Extract one variant's content: ``.content``, a "content" key, or str."""
    if isinstance(item, str):
        return item
    content = _field(item, "content")
    return str(content) if content is not None else None


def _merge_variants(
    record: TrajectoryRecord, extra: tuple[str, ...]
) -> TrajectoryRecord:
    """Return ``record`` with ``extra`` contents appended, deduplicated."""
    merged: list[str] = list(record.variants)
    for content in extra:
        if content not in merged:
            merged.append(content)
    return replace(record, variants=tuple(merged))
