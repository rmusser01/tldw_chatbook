# tldw_chatbook/Agents/run_log_search.py
"""Query the run log: literal by default, structured filters, bounded regex.

Pure module. Literal substring search is the DEFAULT and carries no
line-length cap: `str.__contains__` is linear and cannot backtrack. Regex
is opt-in and scan-bounded: the bound makes catastrophic-backtracking worst
cases finite, not fast. A sufficiently adversarial pattern can still be
expensive within the window. Python's `re` has no match timeout, and
`agent_service._call_with_timeout` abandons rather than kills its worker
thread -- bounding the scan to the first 500 characters avoids runaway
execution, not catastrophe within the bound. See also
`file_operation_tools._MAX_GREP_LINE_SEARCH_CHARS`.

F6 (Qodo #6, PR #1066 review): because `search_run_log` is a RUNTIME tool
(agent_runtime.py dispatches it directly via `deps.search_run_log`, never
through `deps.invoke_tool`), it also bypasses the ordinary per-tool timeout
wrapper (`agent_service._call_with_timeout`) that every catalog tool gets.
The 500-char scan window above bounds the INPUT to one regex evaluation but
does not bound its WORST-CASE TIME -- a single catastrophic-backtracking
match against even 500 characters can still run for a very long time. Two
additional, independently-cheap layers narrow that, and NEITHER is a
complete fix on its own:

  1. A wall-clock deadline (`MAX_SEARCH_SECONDS`) checked between records in
     `search_records`. This bounds the CUMULATIVE cost of scanning many
     records, each individually fast. It CANNOT interrupt a single record
     whose regex evaluation itself hangs -- `re.Pattern.search` is not
     interruptible from pure Python without threads/signals, and this
     module stays synchronous and dependency-free on purpose.
  2. A pattern screen (`_looks_catastrophic`) that rejects the textbook
     nested-quantifier shape (`(a+)+`, `(a*)*`, `(a+)*`, ...) BEFORE
     compiling. This catches the common case that would hang on layer 1's
     very first record, but it is a conservative STRING scan for one known
     dangerous shape, not a general safety proof -- other constructs (e.g.
     alternation-based blowups) can still be slow and are not screened.

Together these make a catastrophic pattern's cost bounded in the common
case and finite in the worst case a wall clock can observe -- they do not
make it fast, and a sufficiently adversarial single-record pattern can
still exceed both bounds before the deadline check next runs. `contains=`
remains the only mode with no such caveat: it is linear and cannot
backtrack by construction, and the tool description tells the model to
prefer it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .run_log_format import RunLogRecord, iter_records

#: Per-record scan window for opt-in regex mode. This bound makes
#: catastrophic-backtracking worst cases finite, not fast. Neither bound
#: makes such a pattern fast — only bounded. Mirrors
#: `file_operation_tools._MAX_GREP_LINE_SEARCH_CHARS`.
MAX_REGEX_SCAN_CHARS = 500

#: F6: wall-clock ceiling on one `search_records` call, checked between
#: records. Deliberately small -- this is meant to be a CHEAP, in-process
#: log search over local files, not a bounded-but-still-slow operation.
MAX_SEARCH_SECONDS = 5.0

#: F6: characters that make a quantified group "already quantified" for the
#: nested-quantifier screen below (`+`/`*`; `?` after either is a lazy
#: variant of the same shape, handled separately).
_QUANTIFIER_CHARS = ("+", "*")


class RunLogSearchTimeout(Exception):
    """A `search_records` call exceeded `MAX_SEARCH_SECONDS`. See F6."""


class RunLogSearchPatternRejected(Exception):
    """`pattern=` matched a known catastrophic-backtracking shape. See F6."""


def _looks_catastrophic(pattern: str) -> bool:
    """Cheap, conservative screen for the classic nested-quantifier shape.

    Detects a parenthesised group whose content ends in a quantifier
    (``+``/``*``, optionally followed by a lazy ``?``) immediately followed
    by another quantifier outside the group -- e.g. ``(a+)+``, ``(a*)*``,
    ``(a+)*``, ``(a*)+``, ``(a+){2,}``. This is THE textbook
    catastrophic-backtracking signature. Deliberately a balanced-paren
    STRING scan, not a regex: a regex screen for dangerous regexes would
    itself need to be immune to the same class of attack. Conservative by
    design -- it can miss more exotic catastrophic shapes (alternation-based
    blowups, deeply nested cross-group cases) but is built to never flag an
    ordinary pattern like ``(abc)+``, ``a+b*``, or ``(foo|bar)+``.

    Args:
        pattern: The model-supplied regex source, unmodified.

    Returns:
        ``True`` when the nested-quantifier shape is found, else ``False``.
    """
    stack: list[int] = []
    n = len(pattern)
    i = 0
    while i < n:
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            start = stack.pop()
            inner_end = i - 1
            if inner_end > start:
                trailing = pattern[inner_end]
                # A lazy quantifier (`+?`/`*?`) is the same dangerous shape
                # one character further back.
                if trailing == "?" and inner_end - 1 > start:
                    trailing = pattern[inner_end - 1]
                if trailing in _QUANTIFIER_CHARS:
                    j = i + 1
                    if j < n and (pattern[j] in _QUANTIFIER_CHARS or pattern[j] == "{"):
                        return True
        i += 1
    return False


def load_records(
    log_dir: Path, *, deadline_seconds: float | None = None
) -> list[RunLogRecord]:
    """Load every complete record from every segment, in order.

    Segment discovery is glob + sort, never the MANIFEST: a crashed run
    writes no manifest, and those are exactly the runs worth inspecting.

    task-1273 review finding B: reading segment files is unbounded I/O --
    a run with several large, multi-segment logs could previously take
    however long that takes with no ceiling at all. `deadline_seconds`
    (optional, `None` by default) makes this bounded the SAME way
    `search_records` bounds its own scan: checked BETWEEN whole units of
    work (segment files here; records there), never mid-read of a single
    segment, which is one blocking I/O call that cannot be interrupted
    partway through without threads. `None` (every call before this task,
    including `search_run_log`'s own `scope="run"` path) preserves prior
    behavior exactly -- unbounded, no check performed at all.

    Args:
        log_dir: The run's log directory.
        deadline_seconds: Optional wall-clock ceiling for this call. When
            set and exceeded before every segment has been read, raises
            rather than returning a partial record list silently -- a
            caller (`search_across_runs`) that went on to search a
            partially-loaded log would look identical to one that searched
            the WHOLE log and found nothing, exactly the "no matches when
            more exists" failure mode task-1273 exists to prevent.

    Returns:
        All records in record-number order; empty when the directory (or a
        segment within it) is unreadable -- an `OSError` degrades to
        "return what was loaded so far", unrelated to `deadline_seconds`.

    Raises:
        RunLogSearchTimeout: `deadline_seconds` is set and was exceeded
            before every segment was read. Some records may be unread; the
            caller must not treat this as "searched, found nothing".
    """
    records: list[RunLogRecord] = []
    started = time.monotonic()
    try:
        for segment in sorted(log_dir.glob("logs.*.txt")):
            if (
                deadline_seconds is not None
                and time.monotonic() - started > deadline_seconds
            ):
                raise RunLogSearchTimeout(
                    f"load_records exceeded its {deadline_seconds:g}s "
                    f"wall-clock budget after reading {len(records)} "
                    f"record(s) from {log_dir} -- further segments were "
                    f"left unread."
                )
            records.extend(iter_records(segment.read_bytes()))
    except OSError:
        return records
    return sorted(records, key=lambda r: r.number)


def search_records(
    records: list[RunLogRecord],
    *,
    contains: str = "",
    pattern: str = "",
    tool: str = "",
    type: str = "",
    status: str = "",
    kind: str = "",
    from_record: int = 0,
    to_record: int = 0,
    context: int = 0,
    limit: int = 50,
    deadline_seconds: float = MAX_SEARCH_SECONDS,
) -> list[RunLogRecord]:
    """Filter ``records``; return hits plus optional neighbouring context.

    Args:
        records: All loaded records, in order.
        contains: Literal substring (case-insensitive). Never compiled.
        pattern: Opt-in regex, searched only over the first
            ``MAX_REGEX_SCAN_CHARS`` characters of each record. Rejected
            up front (``RunLogSearchPatternRejected``) when it matches the
            classic nested-quantifier catastrophic-backtracking shape; see
            the module docstring for why this is a partial, not complete,
            defense.
        tool: Exact tool-name filter.
        type: Exact record-type filter.
        status: Exact status filter.
        kind: Exact agent-kind filter.
        from_record: Inclusive lower bound on record number.
        to_record: Inclusive upper bound on record number.
        context: Include this many records either side of each hit.
        limit: Maximum number of matching records returned; context records
            are returned in addition to this limit.
        deadline_seconds: Wall-clock ceiling for this call, checked between
            records (F6). Cannot interrupt a single record's regex
            evaluation if THAT hangs; see the module docstring.

    Returns:
        Matching records in record order, deduplicated, with context records
        included (result may exceed ``limit`` when context is used).

    Raises:
        RunLogSearchPatternRejected: ``pattern`` matches a known
            catastrophic-backtracking shape.
        RunLogSearchTimeout: the scan exceeded ``deadline_seconds``.
    """
    compiled = None
    if pattern:
        if _looks_catastrophic(pattern):
            raise RunLogSearchPatternRejected(
                f"pattern {pattern!r} looks like it could backtrack "
                f"catastrophically (a quantifier applied to an "
                f"already-quantified group, e.g. (a+)+, (a*)*, (a+)*). "
                f"Use contains=<literal substring> instead -- it is "
                f"unbounded and cannot backtrack."
            )
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []
    needle = contains.lower()
    hit_indexes: list[int] = []
    started = time.monotonic()
    for index, record in enumerate(records):
        if time.monotonic() - started > deadline_seconds:
            raise RunLogSearchTimeout(
                f"search exceeded its {deadline_seconds:g}s wall-clock "
                f"budget after scanning {index} of {len(records)} records. "
                f"Narrow the query -- add 'contains', or filter by 'tool', "
                f"'type', 'from_record'/'to_record' -- and try again."
            )
        if from_record and record.number < from_record:
            continue
        if to_record and record.number > to_record:
            continue
        if tool and record.tool != tool:
            continue
        if type and record.type != type:
            continue
        if status and record.status != status:
            continue
        if kind and record.kind != kind:
            continue
        if needle and needle not in record.content.lower():
            continue
        if compiled is not None and not compiled.search(
            record.content[:MAX_REGEX_SCAN_CHARS]
        ):
            continue
        hit_indexes.append(index)
    # Apply limit to hits first, then expand context around the limited hits.
    # Context records are returned in addition to the limit.
    limited_hit_indexes = hit_indexes[:limit]
    selected: set[int] = set()
    # A negative context would make low > high below, so range(low, high+1)
    # comes back empty and even the hit itself is dropped -- a caller (or a
    # model guessing at search_run_log's args) passing context=-5 would be
    # told "No matching records." even though a match exists, which is
    # worse than an error. Clamp here, at the point of use.
    context = max(0, context)
    for index in limited_hit_indexes:
        low = max(0, index - context)
        high = min(len(records) - 1, index + context)
        selected.update(range(low, high + 1))
    return [records[i] for i in sorted(selected)]


def format_results(
    records: list[RunLogRecord],
    *,
    max_chars: int = 400,
    contains: str = "",
    pattern: str = "",
    offset: int = 0,
) -> str:
    """Render results for the model.

    Each record is rendered as a single block, windowed to at most
    ``max_chars`` characters of its content. Where that window starts is
    decided in this order:

    1. ``offset`` > 0 -- explicit paging always wins. This is how a caller
       reaches content in a record larger than ``max_chars``: the previous
       call's rendered block states the next ``offset`` to pass.
    2. ``contains`` or ``pattern`` has a match in this record -- the window
       is centred on that record's *first* match so the match is always
       inside the rendered text. Before this, rendering always started at
       0, so a record that matched a query could still render a body that
       did not contain the match (TASK-1250).
    3. Neither applies -- the window starts at 0, same as before.

    When the window does not cover the whole record, the block says so:
    the character range shown, the record's total size, and the ``offset``
    to pass to continue reading. Before this, a partial render was silent,
    which is what let a model conclude matched content did not exist.

    F7 (Qodo #7): when a record's ``truncated_from`` is set (the WRITER
    itself capped it at ``run_log_max_record_bytes``), the block also says
    so explicitly -- that content beyond the per-record storage cap was
    never written and cannot be recovered, distinct from the windowing note
    above (which is about what THIS render shows, not what the log stored).

    Args:
        records: Records to render.
        max_chars: Per-record content ceiling in the rendering.
        contains: The literal substring the caller searched for, if any.
            Used only to locate a match to centre the window on here --
            this never re-filters ``records``.
        pattern: The caller's opt-in regex, if any. Matched over only the
            first ``MAX_REGEX_SCAN_CHARS`` characters of a record's
            content, mirroring ``search_records``'s own match decision. An
            invalid pattern is treated as "no match" rather than raising.
        offset: Character offset into each record's content to start
            rendering from, for deterministic paging. Coerced defensively:
            a negative value clamps to 0, and an offset at or past a
            record's end clamps to that record's final window rather than
            rendering nothing.

    Returns:
        One block per record, or a plain no-matches line.
    """
    if not records:
        return "No matching records."
    # A negative offset (a caller, or a model guessing at search_run_log's
    # args, could send one) must not raise or flip the window backwards --
    # clamp at the point of use, same rationale as `context` above.
    offset = max(0, offset)
    compiled = None
    if pattern:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            compiled = None
    blocks = []
    for record in records:
        content = record.content
        total = len(content)
        start = _window_start(
            total=total,
            max_chars=max_chars,
            offset=offset,
            match_pos=_find_match_start(content, contains=contains, compiled=compiled),
        )
        end = min(total, start + max_chars) if max_chars > 0 else start
        body = content[start:end]
        header = (
            f"record {record.number:06d} [{record.type}"
            f"{'/' + record.tool if record.tool and record.tool != '-' else ''}"
            f"{'/' + record.status if record.status and record.status != '-' else ''}]"
        )
        if start > 0 or end < total:
            continuation = f" Use offset={end} to continue." if end < total else ""
            body = f"{body}\n[showing chars {start}-{end} of {total} total.{continuation}]"
        if record.truncated_from:
            # F7 (Qodo #7): the writer caps any record over
            # `run_log_max_record_bytes` and records the ORIGINAL size in
            # `truncated_from` -- but this field was never rendered here, so
            # a model recovering a capped record via search_run_log had no
            # way to learn its tail was unrecoverable and could easily
            # mistake the stored length for the whole result. Byte-accurate
            # (not `total`, which is character length): `truncated_from` is
            # UTF-8 bytes (run_log_format.py's own unit), and the two only
            # coincide for pure-ASCII content.
            stored_bytes = len(record.content.encode("utf-8"))
            body = (
                f"{body}\n[NOTE: this run's log could only store "
                f"{stored_bytes} of this record's original "
                f"{record.truncated_from} bytes (the per-record storage "
                f"cap) -- the remaining "
                f"{record.truncated_from - stored_bytes} bytes were never "
                f"written and cannot be recovered from this run's log.]"
            )
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def _find_match_start(
    content: str, *, contains: str, compiled: re.Pattern[str] | None
) -> int | None:
    """Locate a record's first match, to centre its render window on.

    ``contains`` is searched unbounded, over the record's whole content,
    exactly like ``search_records``' own literal match. ``pattern`` is
    searched only over the first ``MAX_REGEX_SCAN_CHARS`` characters, also
    mirroring ``search_records``' match decision -- the render window is a
    separate concern from that scan bound, so a match found within it is
    still rendered correctly.

    Args:
        content: One record's full content.
        contains: Literal substring (case-insensitive), or empty for none.
        compiled: Compiled opt-in regex, or ``None`` for none.

    Returns:
        The character index of the first match, or ``None`` when neither
        query is set or neither matches this record.
    """
    if contains:
        idx = content.lower().find(contains.lower())
        if idx != -1:
            return idx
    if compiled is not None:
        match = compiled.search(content[:MAX_REGEX_SCAN_CHARS])
        if match is not None:
            return match.start()
    return None


def _window_start(*, total: int, max_chars: int, offset: int, match_pos: int | None) -> int:
    """Pick the character index a record's rendered window starts at.

    Args:
        total: The record's total content length.
        max_chars: The render window's width.
        offset: Explicit, already-clamped-to-non-negative paging offset.
            Wins over ``match_pos`` whenever it is set -- see
            ``format_results``.
        match_pos: The record's first query match position, or ``None``.

    Returns:
        A start index in ``[0, max(0, total - max_chars)]`` (when
        ``max_chars`` <= 0, always 0), so the window never runs past the
        end of ``content`` and, when a window narrower than the whole
        record exists, always contains ``match_pos`` if one was given.
    """
    max_start = max(0, total - max_chars) if max_chars > 0 else 0
    if offset > 0:
        # An offset past the end must still show something rather than an
        # empty window -- clamp to the record's final window instead.
        return min(offset, max_start) if total > max_chars else 0
    if match_pos is None:
        return 0
    # Centre the window on the match, then clamp into bounds. Clamping can
    # only move the window towards the match (see the module docstring's
    # reasoning mirrored here): if centring would run past either edge,
    # clamping to `max_start` or 0 still leaves the match inside the
    # window because max_chars >= max(match_pos - start, 0) + 1 in either
    # clamped case.
    half = max_chars // 2
    start = max(0, match_pos - half)
    return min(start, max_start)


# == Phase 2: aggregation (`run_log_stats`) and slicing (`run_log_slice`) ====
#
# Design spec §10 phase table + task-1271. Both tools share this module's
# central guarantee -- pure functions over an already-loaded record list,
# no filesystem/I/O of their own -- and both are bounded the same way
# `search_records`/`format_results` already are: by a caller-controlled
# constant, never by the log's own size. `agent_service.py` wires the
# `load_records(log_dir)` call and the run's own `max_tool_result_chars`
# ceiling, exactly like it already does for `search_run_log`.

#: `run_log_stats`' `group_by` is restricted to these four metadata fields
#: ON PURPOSE: `type`/`status`/`kind` each have a small, LOG-INDEPENDENT
#: number of distinct values (four record types, a few statuses, two
#: kinds) -- `tool` does NOT (tool names come from the model and from MCP
#: servers, a set this module neither defines nor controls; see
#: `MAX_STATS_GROUPS` below for how THAT case stays bounded). Grouping by
#: anything per-record-unique -- a record number, a `call_id` -- would turn
#: an aggregation tool into an unbounded per-record dump wearing a
#: different schema; that is exactly the shape task-1271 forbids ("no call
#: may return output that scales with log size").
STATS_GROUP_BY_FIELDS = ("tool", "type", "status", "kind")

#: `run_log_slice`'s default window width when the caller omits `to_record`.
DEFAULT_SLICE_WIDTH = 20

#: `run_log_slice`'s hard cap on records returned by ONE call, regardless of
#: how wide `[from_record, to_record]` is or how large the log has grown.
#: Matches `search_records`' own default `limit` so the two tools' worst-
#: case per-call cost stays comparable.
MAX_SLICE_RECORDS = 50

#: `run_log_stats`' hard cap on the number of GROUPS rendered by ONE call.
#: `STATS_GROUP_BY_FIELDS`' own comment above claims each field has a
#: "small, LOG-INDEPENDENT number of distinct values" -- true for `type`/
#: `status`/`kind` (fixed, finite vocabularies this module defines), but
#: NOT true for `tool`: tool names come from whatever the model or an MCP
#: server calls something, a set this module does not control and cannot
#: bound. Without this cap, `compute_stats`' whole justification --
#: "output scales with distinct GROUPS, never with record count" -- breaks
#: exactly when grouping by `tool` on a run that has touched many
#: differently-named tools: the output would again scale with something
#: unbounded, right back to the shape `run_log_stats` exists to avoid.
#: Matches `search_records`' own default `limit` and `MAX_SLICE_RECORDS`
#: so all three run-log tools share one per-call worst-case size ceiling.
MAX_STATS_GROUPS = 50


@dataclass(frozen=True)
class RunLogGroupStats:
    """One group's aggregate counters, as returned by ``compute_stats``."""

    key: str
    count: int
    error_count: int
    content_bytes: int


def compute_stats(
    records: list[RunLogRecord],
    *,
    group_by: str = "tool",
    tool: str = "",
    type: str = "",
    status: str = "",
    kind: str = "",
    from_record: int = 0,
    to_record: int = 0,
    max_groups: int = MAX_STATS_GROUPS,
) -> tuple[list[RunLogGroupStats], int, int]:
    """Aggregate ``records`` into per-group counts, error counts, and bytes.

    The result is O(distinct group values) RENDERED, never O(records) --
    but distinct group values are only bounded by construction for
    ``type``/``status``/``kind`` (small, fixed vocabularies this module
    defines). Grouping by ``tool`` has no such bound: tool names come from
    whatever the model or an MCP server calls something. ``max_groups``
    is what actually keeps the RETURNED value bounded in that case -- the
    top ``max_groups`` groups by count are kept, and the rest are counted
    but not rendered (see ``omitted_group_count`` below). This is what
    lets ``run_log_stats`` answer "which tool have I called most, and how
    often did it fail?" without paging the log itself through the model's
    context: the whole log is scanned here (same O(records) cost
    ``search_records`` already pays), but what comes BACK is capped
    regardless of how many distinct tool names -- or records -- the run
    has accumulated.

    No token totals are computed here. `RunLogRecord` does not carry a
    per-record token count -- ``agent_runtime.run_agent_loop`` tracks a
    running `total_tokens` in memory and the manifest records the whole
    run's final total once the run ends, but neither is threaded into the
    log's own record format (Phase 1, §4.1, deliberately fixed that
    format). Fabricating a per-group estimate here (e.g. via a tokenizer
    heuristic) would silently disagree with the run's own authoritative
    accounting in `RunOutcome.total_tokens` / the budget check in the
    loop, which is worse than not reporting it. `content_bytes` is the
    honest, exact substitute available from what the log actually stores.

    Args:
        records: All loaded records, in order (as from ``load_records``).
        group_by: One of ``STATS_GROUP_BY_FIELDS``. A value not in that
            set (a model could send anything) falls back to ``"tool"``
            rather than raising -- see the module's defensive-coercion
            convention (``run_log_search.py``'s callers in
            ``agent_service.py``).
        tool: Optional exact tool-name pre-filter, applied before
            grouping. Mirrors ``search_records``'s own filter.
        type: Optional exact record-type pre-filter.
        status: Optional exact status pre-filter.
        kind: Optional exact agent-kind pre-filter (``primary``/
            ``subagent``) -- a primary agent's own log holds its spawned
            children's records too (spec §4.1), so this can answer "how
            much of my log came from sub-agents".
        from_record: Optional inclusive lower bound on record number.
        to_record: Optional inclusive upper bound on record number.
        max_groups: Hard cap on how many groups are RETURNED, regardless of
            how many distinct ``group_by`` values the (filtered) records
            actually contain. This is the boundedness guarantee for the
            unbounded case (``group_by="tool"``): a run that has called
            hundreds of differently-named tools still gets at most
            ``max_groups`` rows back, not one row per tool name.

    Returns:
        A 3-tuple ``(groups, total_matched, omitted_group_count)``:

        - ``groups``: at most ``max_groups`` ``RunLogGroupStats``, one per
          distinct group value present after filtering, sorted by
          descending ``count`` (ties broken alphabetically by ``key``) so
          "which tool have I called most" is always the first row AND
          always survives the cap. Empty when no record matches.
        - ``total_matched``: how many records matched the pre-filters,
          across ALL groups -- including any cut by ``max_groups`` -- so a
          caller can report an accurate record total independent of the
          group cap.
        - ``omitted_group_count``: how many additional distinct groups
          existed but were cut to enforce ``max_groups`` -- 0 when every
          group fit. A caller MUST report this explicitly rather than
          silently dropping it: a truncated statistic presented as
          complete is worse than no statistic, because the reader has no
          way to tell the two apart.
    """
    if group_by not in STATS_GROUP_BY_FIELDS:
        group_by = "tool"
    # key -> [count, error_count, content_bytes]; a plain dict of lists
    # avoids importing collections.Counter/defaultdict for three counters.
    groups: dict[str, list[int]] = {}
    total_matched = 0
    for record in records:
        if from_record and record.number < from_record:
            continue
        if to_record and record.number > to_record:
            continue
        if tool and record.tool != tool:
            continue
        if type and record.type != type:
            continue
        if status and record.status != status:
            continue
        if kind and record.kind != kind:
            continue
        total_matched += 1
        key = getattr(record, group_by) or "-"
        bucket = groups.setdefault(key, [0, 0, 0])
        bucket[0] += 1
        if record.status == "error":
            bucket[1] += 1
        bucket[2] += len(record.content.encode("utf-8"))
    all_groups = sorted(
        (
            RunLogGroupStats(key=key, count=c, error_count=e, content_bytes=b)
            for key, (c, e, b) in groups.items()
        ),
        key=lambda g: (-g.count, g.key),
    )
    if max_groups > 0 and len(all_groups) > max_groups:
        return all_groups[:max_groups], total_matched, len(all_groups) - max_groups
    return all_groups, total_matched, 0


def format_stats(
    groups: list[RunLogGroupStats],
    *,
    group_by: str,
    total_records: int,
    omitted_groups: int = 0,
) -> str:
    """Render ``compute_stats``' result for the model.

    Args:
        groups: ``compute_stats``'s (already ``max_groups``-capped) group
            list.
        group_by: The dimension grouped on (after its own fallback), named
            in the header line.
        total_records: Count of records considered after any pre-filters,
            before grouping -- ``compute_stats``'s own ``total_matched``.
            Reported separately, and independent of any group cap, so the
            model always sees an accurate record count even when
            ``omitted_groups`` is nonzero (it is NOT guaranteed to equal
            ``sum(g.count for g in groups)`` in that case, precisely
            because some groups' counts are not in ``groups`` at all).
        omitted_groups: ``compute_stats``'s own ``omitted_group_count``.
            When nonzero, an explicit trailer line reports how many
            further distinct ``group_by`` values exist beyond the ones
            rendered -- a capped result MUST say so rather than silently
            presenting a partial group list as if it were complete.

    Returns:
        A header line, one line per (capped) group, and -- only when
        ``omitted_groups`` is nonzero -- one trailer line naming the
        omitted count. Bounded by ``max_groups``, never by the number of
        records. A plain "No records matched." line when ``groups`` is
        empty.
    """
    if not groups:
        return "No records matched."
    lines = [f"{total_records} record(s) in this run's log, grouped by {group_by}:"]
    for g in groups:
        lines.append(
            f"  {g.key}: count={g.count} errors={g.error_count} "
            f"content_bytes={g.content_bytes}"
        )
    if omitted_groups:
        lines.append(
            f"  ... and {omitted_groups} further distinct {group_by} value(s) "
            "omitted (showing the most frequent groups only; narrow with "
            "tool=/type=/status=/kind= or from_record=/to_record= to see "
            "the rest)."
        )
    return "\n".join(lines)


def slice_records(
    records: list[RunLogRecord],
    *,
    from_record: int,
    to_record: int = 0,
    max_records: int = MAX_SLICE_RECORDS,
) -> tuple[list[RunLogRecord], int, int, int]:
    """Select a contiguous, count-capped range of records by record number.

    Unlike ``search_records``, this never scans content -- it is a pure
    range selection on ``record.number``, which is what makes "retrieve a
    coherent stretch of my own reasoning" cheap and predictable rather
    than an accidental content search with an empty query.

    Args:
        records: All loaded records, in order (as from ``load_records``).
        from_record: Inclusive lower bound on record number. Coerced to at
            least 1 (a caller could pass 0 or a negative number).
        to_record: Inclusive upper bound. 0 (the default, meaning "not
            given") resolves to ``from_record + DEFAULT_SLICE_WIDTH - 1``.
            A value below the resolved ``from_record`` resolves to
            ``from_record`` itself (a one-record slice) rather than an
            empty range or an error.
        max_records: Hard cap on how many records are RETURNED, regardless
            of how wide ``[from_record, to_record]`` is. This is the
            boundedness guarantee: a model requesting
            ``from_record=1, to_record=999999`` on a long run gets
            ``max_records`` records back, not the whole log.

    Returns:
        A 4-tuple ``(selected, total_matched, resolved_from,
        resolved_to)``:

        - ``selected``: the records to render, in order, capped at
          ``max_records`` (taken from the LOW end of the range, so paging
          forward from a returned ``to_record + 1`` always makes
          progress).
        - ``total_matched``: how many records actually fell in
          ``[resolved_from, resolved_to]`` before the cap was applied --
          lets a caller report when it clipped (``total_matched >
          len(selected)``).
        - ``resolved_from`` / ``resolved_to``: the bounds actually used,
          after defaulting/clamping the raw input -- reported even when
          ``selected`` is empty, so a caller can still say what range was
          searched.
    """
    resolved_from = max(1, from_record)
    resolved_to = to_record if to_record > 0 else resolved_from + DEFAULT_SLICE_WIDTH - 1
    if resolved_to < resolved_from:
        resolved_to = resolved_from
    matched = [r for r in records if resolved_from <= r.number <= resolved_to]
    return matched[:max_records], len(matched), resolved_from, resolved_to


def format_slice(
    records: list[RunLogRecord],
    *,
    from_record: int,
    to_record: int,
    total_matched: int,
    max_chars: int = 400,
) -> str:
    """Render a contiguous record range as one coherent block.

    Deliberately reuses ``format_results`` for the per-record rendering
    rather than writing a second one: ``run_log_slice`` must bound its
    output exactly the way ``search_run_log`` bounds its own (task-1271),
    and a second renderer would be a second place for that bound -- and
    the TASK-1250 "match not in the rendered window" class of bug -- to
    drift out of sync. No ``contains``/``pattern`` is passed through here:
    a slice has no query to centre a window on, so every record renders
    from its own start, same as a plain (query-less) ``format_results``
    call already does.

    Args:
        records: The already range-selected and count-capped records to
            render, in order -- normally ``slice_records``'s ``selected``
            return value.
        from_record: The RESOLVED lower bound that was searched (i.e.
            ``slice_records``'s ``resolved_from``, not the caller's raw,
            possibly-defaulted input) -- used to report the range in the
            header, including when ``records`` is empty.
        to_record: The resolved upper bound, same provenance.
        total_matched: ``slice_records``'s own ``total_matched`` -- lets
            this note when the range was wider than what got returned.
        max_chars: Per-record content ceiling, forwarded to
            ``format_results`` unchanged.

    Returns:
        A header line stating the record-number range covered (plus a
        clipping note when ``total_matched`` exceeds what was returned)
        followed by ``format_results``'s rendering -- or a plain "No
        records numbered ..." line when ``records`` is empty, so an empty
        slice reads distinctly from "no search hits" (``format_results``'
        own empty-input message) instead of looking identical to it.
    """
    if not records:
        return f"No records numbered {from_record:06d}-{to_record:06d} in this run's log."
    lo, hi = records[0].number, records[-1].number
    header = f"records {lo:06d}-{hi:06d} of this run's log"
    if total_matched > len(records):
        header += (
            f" (showing {len(records)} of {total_matched} records in the "
            f"requested range {from_record:06d}-{to_record:06d}; continue "
            f"with from_record={hi + 1} for the rest)"
        )
    return f"{header}:\n\n{format_results(records, max_chars=max_chars)}"


# == Cross-run search (task-1273) =============================================
#
# task-1271 deferred this rather than build it against a guess; the deferral
# (task-1273) found the gap narrower than first thought: `AgentRunsDB.
# list_runs(conversation_id)` already enumerates a conversation's runs, and
# `run_log.resolve_existing_log_dir(run_id)` (TASK-870) already locates an
# arbitrary run's log directory by id, read-only. Composing those two gives
# best-effort cross-run search without a schema change -- option (a) of the
# task's two honest choices; option (b) (recording a run's resolved root at
# write time, for the runs this can never find) is a separate, later
# decision that does not block this one.
#
# THE ONE HONEST LIMITATION: nothing records which ROOT an older run's log
# was written under, so a run whose log is not reachable under the CURRENT
# root (the workspace folder was bound, rebound, or unbound since) cannot be
# found here -- `resolve_existing_log_dir` correctly returns `None` for it.
# That must never read as "there is nothing there": `CrossRunSearchResult`
# and `format_cross_run_results` below report coverage explicitly --
# searched vs. unresolved vs. never attempted -- rather than only surfacing
# whatever hits happened to be found. Silently dropping that count is
# exactly the "no matches in 3 earlier runs when 8 exist" failure mode this
# task exists to avoid.
#
# This section stays a pure, filesystem-explicit sibling of `load_records`/
# `search_records` above: it is handed already-resolved `(run_id, log_dir)`
# pairs (or `None` for an unresolved log) rather than resolving roots or
# querying a DB itself -- `agent_service.py`'s `search_run_log` closure
# (the only impure caller) does that resolution, exactly like it already
# resolves ONE `log_dir` today before calling `load_records`.

#: `search_run_log`'s `scope="conversation"` mode hard cap on how many of a
#: conversation's PRIMARY runs (the current run included) are scanned in
#: ONE call, regardless of how many runs the conversation has accumulated.
#: Mirrors `MAX_STATS_GROUPS`/`MAX_SLICE_RECORDS`: bounds this call's own
#: work, and -- together with the shared hit `limit` and shared wall-clock
#: `deadline_seconds` `search_across_runs` applies below -- its OUTPUT,
#: never the conversation's run count.
MAX_CROSS_RUN_RUNS = 10


@dataclass(frozen=True)
class CrossRunHit:
    """One `scope="conversation"` search hit, tagged with its source run.

    `record.run_id` (see `run_log_format.RunLogRecord`) already names
    whichever run actually WROTE the record -- a primary run's own turn, or
    one of its sub-agents' (a child run never gets a log directory of its
    own; its records land in its PARENT's directory -- see
    `RunLogWriter.bind`'s docstring). `source_run_id` instead names the
    PRIMARY run whose log DIRECTORY this hit was found under -- the
    granularity `AgentRunsDB.list_runs` enumerates and this module reports
    coverage at. The two coincide for a primary run's own record and differ
    only for a record written by one of that primary run's sub-agents.
    """

    record: RunLogRecord
    source_run_id: str
    is_current_run: bool


@dataclass(frozen=True)
class CrossRunSearchResult:
    """`search_across_runs`'s return value: hits plus an honest coverage report.

    A caller (`format_cross_run_results`) MUST report `unresolved_run_ids`
    and `not_searched_run_ids` explicitly rather than only rendering `hits`
    -- an empty `hits` list must never be presented as "searched and found
    nothing" when some of the runs it was supposed to cover were never
    actually scanned. See the module section header above for why.
    """

    hits: list[CrossRunHit]
    #: Runs whose log was located and scanned (even one that contributed no
    #: hits, e.g. because the shared `limit` was already spent by an
    #: earlier, newer run -- "searched" means scanned, not "produced a hit").
    searched_run_ids: list[str]
    #: Runs whose log could NOT be located under the current root -- the
    #: one honest limitation this module cannot paper over (see above).
    unresolved_run_ids: list[str]
    #: Runs that were never attempted at all: this call's shared wall-clock
    #: `deadline_seconds` ran out before reaching them (`search_across_runs`
    #: itself), or the `MAX_CROSS_RUN_RUNS` cap excluded them before this
    #: function ever saw them (`agent_service.py`'s caller adds those in).
    #: Distinct from `unresolved_run_ids`: these runs' logs may well exist
    #: and be perfectly reachable -- there simply was no room in this call's
    #: bounded budget to check.
    not_searched_run_ids: list[str] = field(default_factory=list)


def search_across_runs(
    resolved_runs: list[tuple[str, Path | None]],
    *,
    current_run_id: str,
    contains: str = "",
    pattern: str = "",
    tool: str = "",
    type: str = "",
    status: str = "",
    kind: str = "",
    from_record: int = 0,
    to_record: int = 0,
    context: int = 0,
    limit: int = 50,
    deadline_seconds: float = MAX_SEARCH_SECONDS,
) -> CrossRunSearchResult:
    """Search every already-resolved run's log as ONE bounded call.

    Applies `search_records`'s own filters identically to each run in
    `resolved_runs`, in order -- but both bounds it shares with
    `search_records` (`limit`, `deadline_seconds`) are SHARED BUDGETS across
    every run combined, not reset per run:

    - `limit`: once enough hits have accumulated across the runs searched
      so far, later runs are still scanned (so `searched_run_ids` stays
      accurate) but contribute no further hits -- the RENDERED output
      cannot grow with the number of runs searched, only with `limit`.
    - `deadline_seconds`: resetting this per run would let one
      `scope="conversation"` call cost `len(resolved_runs) *
      deadline_seconds` in the worst case (a genuinely slow regex against
      every run's log), defeating the "cheap, in-process log search"
      guarantee `MAX_SEARCH_SECONDS` exists for in the single-run case --
      see the module docstring's F6 section. Once the shared budget is
      spent, remaining runs with a locatable log are reported via
      `CrossRunSearchResult.not_searched_run_ids` rather than scanned.
      task-1273 review finding B: this budget covers `load_records` too,
      not only `search_records` -- reading a run's segment files is
      unbounded I/O, and a conversation with several large, multi-segment
      logs could otherwise blow the whole shared budget on loading alone,
      before a single search ran. The remaining deadline is recomputed
      immediately after each `load_records` call (its own I/O is not
      free) and `load_records` itself is called with that remaining
      budget so it can stop reading further segments early rather than
      finishing an over-budget read only to have the result discarded.

    Args:
        resolved_runs: `(run_id, log_dir)` pairs, newest run first,
            already capped by the caller to at most `MAX_CROSS_RUN_RUNS`
            entries (`agent_service.py` resolves `AgentRunsDB.list_runs`'
            own newest-first order through `run_log.resolve_existing_log_dir`
            before calling this -- this function does no DB or root
            resolution of its own, mirroring `load_records`'s own
            explicit-path contract). `log_dir` is `None` for a run whose
            log could not be located under the current root.
        current_run_id: The run this call is executing in, so each hit can
            be labelled "this run" vs. an earlier one.
        contains, pattern, tool, type, status, kind, from_record, to_record,
            context: Forwarded to `search_records` unchanged, applied to
            each run's own records in turn.
        limit: Shared hit budget across every run combined (see above).
        deadline_seconds: Shared wall-clock budget across every run
            combined (see above).

    Returns:
        A `CrossRunSearchResult` naming which runs were actually searched
        (found and scanned, regardless of whether the shared `limit` left
        room for any of their hits), which could not be located, and which
        were never attempted because the shared time budget ran out first.

    Raises:
        RunLogSearchPatternRejected: `pattern` matches a known
            catastrophic-backtracking shape (raised by the first `run_id`'s
            `search_records` call, exactly like the single-run case).
        RunLogSearchTimeout: a single run's own SEARCH (not load -- see
            below) exceeded its share of `deadline_seconds` -- exactly
            like the single-run case, this is NOT caught here; a caller
            degrading gracefully must catch it the same way it already
            catches it for `scope="run"`. A run whose LOADING alone
            exhausts the shared budget is handled differently, entirely
            inside this function (task-1273 review finding B): reading
            segment files is unbounded I/O, so `load_records` is called
            with the remaining budget and, if IT raises this same
            exception, that run is recorded as `not_searched` (its log may
            hold more records than were read; scanning a partial list
            would silently look like a complete search that found
            nothing) and the loop moves on to the next run rather than
            failing the whole call.
    """
    hits: list[CrossRunHit] = []
    searched: list[str] = []
    unresolved: list[str] = []
    not_searched: list[str] = []
    started = time.monotonic()
    remaining_limit = limit
    for run_id, log_dir in resolved_runs:
        if log_dir is None:
            unresolved.append(run_id)
            continue
        remaining_deadline = deadline_seconds - (time.monotonic() - started)
        if remaining_deadline <= 0:
            # Shared wall-clock budget already spent by earlier runs in
            # this same call -- this run's log IS locatable, there was
            # simply no time left to scan it. Never conflated with
            # `unresolved` (see `CrossRunSearchResult`'s own field docs).
            not_searched.append(run_id)
            continue
        try:
            records = load_records(log_dir, deadline_seconds=remaining_deadline)
        except RunLogSearchTimeout:
            # task-1273 review finding B: loading is unbounded I/O that
            # must count against the shared budget too -- a run whose log
            # took the remaining time just to LOAD (before a single search
            # ran) is `not_searched`, never scanned against a partial
            # record list.
            not_searched.append(run_id)
            continue
        # Recompute AFTER loading, before searching: load_records' own I/O
        # is not free, and the shared deadline must reflect time actually
        # spent so far, not the pre-load estimate above.
        remaining_deadline = deadline_seconds - (time.monotonic() - started)
        if remaining_deadline <= 0:
            not_searched.append(run_id)
            continue
        per_run_limit = remaining_limit if remaining_limit > 0 else 0
        found = search_records(
            records,
            contains=contains,
            pattern=pattern,
            tool=tool,
            type=type,
            status=status,
            kind=kind,
            from_record=from_record,
            to_record=to_record,
            context=context,
            limit=per_run_limit,
            deadline_seconds=remaining_deadline,
        )
        searched.append(run_id)
        for record in found:
            hits.append(
                CrossRunHit(
                    record=record,
                    source_run_id=run_id,
                    is_current_run=(run_id == current_run_id),
                )
            )
        if limit > 0:
            remaining_limit = limit - len(hits)
    return CrossRunSearchResult(
        hits=hits,
        searched_run_ids=searched,
        unresolved_run_ids=unresolved,
        not_searched_run_ids=not_searched,
    )


def format_cross_run_results(
    result: CrossRunSearchResult,
    *,
    max_chars: int = 400,
    contains: str = "",
    pattern: str = "",
    offset: int = 0,
    omitted_run_count: int = 0,
) -> str:
    """Render `search_across_runs`' result: coverage, then attributed hits.

    ALWAYS leads with an explicit coverage line -- how many runs were
    actually searched, vs. could not be located, vs. never attempted --
    rather than only rendering `hits`. See `CrossRunSearchResult`'s own
    docstring for why this line is not optional: a caller told "no matches"
    when several of the conversation's runs were never actually searched
    would draw a confident wrong conclusion (task-1273).

    Args:
        result: `search_across_runs`'s return value.
        max_chars: Forwarded to `format_results`, applied per hit.
        contains: The literal substring searched for, forwarded to
            `format_results` to centre each hit's rendered window.
        pattern: The opt-in regex searched for, same purpose.
        offset: Forwarded to `format_results`, applied to every hit's
            rendered content identically -- same semantics as the
            single-run case.
        omitted_run_count: Runs never attempted because they fell outside
            `MAX_CROSS_RUN_RUNS` -- counted separately from
            `result.not_searched_run_ids` (whose exact ids ARE known,
            since `search_across_runs` was handed a fully-resolved list)
            because the caller (`agent_service.py`) deliberately queries
            only an EXACT COUNT of runs beyond the cap (task-1273 review
            finding A) rather than fetching every one just to list their
            ids -- that unbounded fetch is exactly what finding A flagged.
            Folded into the same "not attempted" coverage note as
            `not_searched_run_ids`; both mean the same thing to a reading
            agent (a run that may still exist and be searchable, just not
            reached by this call), only the REASON differs (a cap vs. the
            shared time budget).

    Returns:
        A coverage line, then one attributed block per hit (labelled "this
        run" or "an earlier run (<run_id>)"), or the coverage line plus
        "No matching records." when there were no hits.
    """
    not_attempted = len(result.not_searched_run_ids) + max(0, omitted_run_count)
    total = len(result.searched_run_ids) + len(result.unresolved_run_ids) + not_attempted
    coverage = (
        f"Searched {len(result.searched_run_ids)} of {total} run(s) "
        "in this conversation"
    )
    notes = []
    if result.unresolved_run_ids:
        notes.append(
            f"{len(result.unresolved_run_ids)} could not be located under "
            "the current root and were NOT searched (their log directory "
            "is not reachable -- e.g. the workspace folder was bound, "
            "rebound, or unbound since)"
        )
    if not_attempted:
        notes.append(
            f"{not_attempted} not attempted this call (this call's run or "
            "time budget was reached first)"
        )
    if notes:
        coverage += ": " + "; ".join(notes)
    coverage += "."
    lines = [coverage]
    if not result.hits:
        lines.append("No matching records.")
        return "\n".join(lines)
    for hit in result.hits:
        label = (
            "this run"
            if hit.is_current_run
            else f"an earlier run ({hit.source_run_id})"
        )
        lines.append(
            f"[{label}]\n"
            + format_results(
                [hit.record],
                max_chars=max_chars,
                contains=contains,
                pattern=pattern,
                offset=offset,
            )
        )
    return "\n\n".join(lines)
