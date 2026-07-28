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


def load_records(log_dir: Path) -> list[RunLogRecord]:
    """Load every complete record from every segment, in order.

    Segment discovery is glob + sort, never the MANIFEST: a crashed run
    writes no manifest, and those are exactly the runs worth inspecting.

    Args:
        log_dir: The run's log directory.

    Returns:
        All records in record-number order; empty when unreadable.
    """
    records: list[RunLogRecord] = []
    try:
        for segment in sorted(log_dir.glob("logs.*.txt")):
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
