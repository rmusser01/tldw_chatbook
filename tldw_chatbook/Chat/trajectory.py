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

Never-fabricate contract
    Timing is surfaced exactly as stored. NULL sidecar timing becomes
    ``None`` fields, and tool rows created at marker-append time keep
    their append-time (zero-duration) stamps verbatim -- the projection
    never derives, defaults, or "corrects" a timestamp.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from tldw_chatbook.Chat.provider_usage import ProviderUsage

__all__ = [
    "TrajectoryRecord",
    "TrajectorySnapshot",
    "TrajectoryTurn",
    "derive_trajectory",
]

#: Cap for ``TrajectoryRecord.content_preview`` (spec: first 120 chars).
PREVIEW_MAX_CHARS = 120

#: Ledger ``kind`` values (spec data model).
KIND_USER = "user"
KIND_ASSISTANT = "assistant"
KIND_TOOL_CALL = "tool_call"
KIND_TOOL_RESULT = "tool_result"
KIND_COMPACTION = "compaction"

_TOOL_KINDS = frozenset({KIND_TOOL_CALL, KIND_TOOL_RESULT})
_RENDERED_ROLES = frozenset({KIND_USER, KIND_ASSISTANT})
_COMPACTION_PURPOSE = "conversation_compaction"

# Sentinel for "no sidecar seq" in sort keys; larger than any real seq.
_NO_SEQ = float("inf")


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

    # Sidecar rows: message rows by message id, tool rows by owning id.
    message_rows: dict[str, Any] = {}
    tool_rows: dict[str, list[Any]] = {}
    for row in traj_rows:
        mid = _field(row, "message_id")
        kind = str(_field(row, "event_kind") or "")
        if not mid:
            continue
        if kind in _TOOL_KINDS:
            tool_rows.setdefault(str(mid), []).append(row)
        else:
            message_rows[str(mid)] = row

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
        for tool_row in sorted(
            tool_rows.get(m.mid, ()),
            key=lambda r: _as_int(_field(r, "seq")),
        ):
            events.append(
                (
                    turn_id,
                    _tool_record(tool_row, turn_id=turn_id),
                )
            )

    turns = _group_turns(events)
    turns = _insert_compaction_markers(turns, compaction_records, turn_msg_times)
    turns = _apply_variant_sets(turns, variant_sets)
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
    return TrajectoryRecord(
        seq=seq,
        kind=record.kind,
        turn_id=record.turn_id,
        message_id=record.message_id,
        content_preview=record.content_preview,
        usage=record.usage,
        step_started_at=record.step_started_at,
        first_token_at=record.first_token_at,
        completed_at=record.completed_at,
        model=record.model,
        provider=record.provider,
        payload=record.payload,
        variants=record.variants,
        depth=record.depth,
    )


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
    return TrajectoryRecord(
        seq=0,  # assigned by the final pass
        kind=kind,
        turn_id=turn_id,
        message_id=m.mid,
        content_preview=_preview(m.content),
        usage=usage,
        step_started_at=_field(row, "step_started_at") if row is not None else None,
        first_token_at=_field(row, "first_token_at") if row is not None else None,
        completed_at=_field(row, "completed_at") if row is not None else None,
        model=_field(row, "model") if row is not None else None,
        provider=_field(row, "provider") if row is not None else None,
        payload=None,
        variants=variants,
        depth=0,
    )


def _tool_record(row: Any, *, turn_id: str) -> TrajectoryRecord:
    """Build a depth-1 tool record from a sidecar ``tool_*`` row.

    ``message_id`` is the OWNING assistant message's id (both tool kinds
    key on it). Timing is surfaced exactly as stored -- append-time
    zero-duration stamps stay verbatim. The preview shows the tool name
    and result text (marker convention: ``name -> result``), never the
    raw ``payload_json`` envelope.
    """
    payload = _parse_payload(_field(row, "payload_json"))
    return TrajectoryRecord(
        seq=0,  # assigned by the final pass
        kind=str(_field(row, "event_kind") or ""),
        turn_id=turn_id,
        message_id=str(_field(row, "message_id") or "") or None,
        content_preview=_tool_preview(payload),
        usage=None,
        step_started_at=_field(row, "step_started_at"),
        first_token_at=_field(row, "first_token_at"),
        completed_at=_field(row, "completed_at"),
        model=_field(row, "model"),
        provider=_field(row, "provider"),
        payload=payload,
        variants=(),
        depth=1,
    )


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
        if str(_field(rec, "purpose") or _COMPACTION_PURPOSE)
        == _COMPACTION_PURPOSE
    ]
    if not markers or not turns:
        return turns

    markers = sorted(
        enumerate(markers),
        key=lambda pair: (_parse_timestamp(_field(pair[1], "started_at")) or 0.0, pair[0]),
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
            records.extend(_marker_record(rec, turn_id=turn.turn_id) for rec in lead)
        records.extend(turn.records)
        records.extend(_marker_record(rec, turn_id=turn.turn_id) for rec in buckets[i])
        result.append(TrajectoryTurn(turn.turn_id, tuple(records)))
    return result


def _marker_record(rec: Any, *, turn_id: str) -> TrajectoryRecord:
    """Build a between-turn compaction marker record."""
    status = str(_field(rec, "status") or "")
    usage = ProviderUsage.from_json(_field(rec, "provider_usage_json"))
    return TrajectoryRecord(
        seq=0,  # assigned by the final pass
        kind=KIND_COMPACTION,
        turn_id=turn_id,
        message_id=None,
        content_preview=f"compaction: {status}" if status else "compaction",
        usage=usage,
        step_started_at=_parse_timestamp(_field(rec, "started_at")),
        first_token_at=None,  # compaction has no token boundary
        completed_at=_parse_timestamp(_field(rec, "finished_at")),
        model=_field(rec, "model") or None,
        provider=_field(rec, "provider") or None,
        payload=None,
        variants=(),
        depth=0,
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
        superseded_by_turn[str(turn_id)] = (
            superseded_by_turn[str(turn_id)] + contents
        )

    if not superseded_by_turn:
        return turns

    result: list[TrajectoryTurn] = []
    for turn in turns:
        extra = superseded_by_turn.get(turn.turn_id)
        if not extra:
            result.append(turn)
            continue
        records = tuple(
            _merge_variants(record, extra)
            if record.kind == KIND_ASSISTANT
            else record
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
    return TrajectoryRecord(
        seq=record.seq,
        kind=record.kind,
        turn_id=record.turn_id,
        message_id=record.message_id,
        content_preview=record.content_preview,
        usage=record.usage,
        step_started_at=record.step_started_at,
        first_token_at=record.first_token_at,
        completed_at=record.completed_at,
        model=record.model,
        provider=record.provider,
        payload=record.payload,
        variants=tuple(merged),
        depth=record.depth,
    )
